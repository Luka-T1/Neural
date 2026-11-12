"""Organizam: LIF neuroni sa provodnim sinapsama (nS, mV, pA, ms; mirovanje -65 mV).

CONFIDENCE = 'speculative'  (lokalna privremena implementacija; nije zamena za
neural.mechanisms; vidi docs/traceability.md)

Organizam prima samo vektor (9,) struja u nA i vraća vektor (4,) premotornih struja u nA.
Ponašanje sme da izranja samo iz dinamike: nema grananja po vrednostima pojedinačnih senzora.

Plastičnost: tri pravila `neural.mechanisms.da_stdp.DASTDP` (self.rules): 'sens_exc'
(senzorni -> E), 'exc_cpg' (E -> CPG) i 'cross' (samo A<->B blokovi CPG-a; maska je 0 unutar
A-A i B-B). Težine su normalizovane na w_max_nS = w_max_factor * mean(početnih plastičnih težina),
pa je w u [0, 1]. DA pojačanje g_da = eta*1e-3 (1/ms); tonični d = g_da*da_tonic, a po koraku se
ubacuje 'da' = g_da*da_phasic*dt/tau_d, pa je u stacionarnom stanju d = g_da*(tonic+phasic).
Ažuriranje ide svakog koraka (redosled raspad -> spajk -> trag, Δt = t_post - t_pre).

CPG ostaje lokalni spiking half-center, a NE neural.mechanisms.cpg.MatsuokaCPG: Matsuoka je
model brzine bez spajkova, pa DA-STDP na A<->B sinapsama za njega ne bi bio definisan
(nema pre/post spajkova za tragove podobnosti).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from neural.mechanisms.da_stdp import DASTDP

CONFIDENCE = "speculative"
PARAMS_META = {
    "tau_ampa_ms": {"value": 2.0, "unit": "ms", "source": "Destexhe1998", "confidence": "strongly_supported"},
    "tau_gaba_ms": {"value": 6.0, "unit": "ms", "source": "Destexhe1998", "confidence": "strongly_supported"},
    "E_gaba_mV": {"value": -75.0, "unit": "mV", "source": "ASSUMPTION: zadato u specifikaciji zadatka", "confidence": "incomplete"},
    "tau_c_ms": {"value": 1000.0, "unit": "ms", "source": "Izhikevich2007", "confidence": "hypothesis"},
    "tau_stdp_ms": {"value": 20.0, "unit": "ms", "source": "BiPoo2001; ASSUMPTION: zaokruženo na 20 ms", "confidence": "strongly_supported"},
    "cpg_topology": {"value": "half-center", "unit": "-", "source": "Brown1911; Matsuoka1985; MarderCalabrese1996", "confidence": "strongly_supported"},
    "g_da_per_ms": {"value": 2e-5, "unit": "1/ms", "source": "ASSUMPTION: g_da = eta*1e-3 je ekvivalent ranijeg eta*w_max*c*DA pri normalizovanim težinama", "confidence": "speculative"},
    "w_max_nS": {"value": "w_max_factor*mean(w0)", "unit": "nS", "source": "ASSUMPTION: gornja granica relativna prema srednjoj početnoj težini; težine u DASTDP su w/w_max_nS", "confidence": "speculative"},
    "tau_d_ms": {"value": 200.0, "unit": "ms", "source": "Izhikevich2007; ASSUMPTION: podrazumevano tau_d iz DASTDP", "confidence": "hypothesis"},
    "cross_mask": {"value": "A<->B only", "unit": "-", "source": "ASSUMPTION: unakrsne plastične veze samo između A i B populacija CPG-a", "confidence": "speculative"},
    "tau_w_ms": {"value": "300-600", "unit": "ms", "source": "ASSUMPTION: adaptacija koja daje ~1 Hz", "confidence": "hypothesis"},
}

TAU_D_MS = 200.0
N_SENS, N_ACT = 9, 4
PER_SENS = 5


@dataclass
class OrganismConfig:
    # ćelija
    C_pF: float = 200.0
    gL_nS: float = 10.0
    E_L: float = -65.0
    V_th: float = -50.0
    V_reset: float = -65.0
    t_ref_ms: float = 2.0
    # sinapse
    tau_ampa_ms: float = 2.0
    tau_gaba_ms: float = 6.0
    E_ampa: float = 0.0
    E_gaba: float = -75.0
    # populacije
    n_exc: int = 80
    n_inh: int = 20
    p_conn: float = 0.1
    cpg_pop_size: int = 10
    cpg_inh_size: int = 5
    tau_w_ms: float = 450.0          # sredina opsega 300-600 ms
    w_adapt_inc_pA: float = 8.0
    # pogon i šum (nA)
    sensory_gain: float = 1.0
    sensory_bias_nA: float = 0.12
    bias_exc_nA: float = 0.10
    bias_inh_nA: float = 0.05
    bias_cpg_nA: float = 0.28
    noise_nA: float = 0.03
    # težine (nS)
    w_sens_exc: float = 0.8
    w_ee: float = 0.5
    w_ei: float = 0.8
    w_ie: float = 2.0
    w_ii: float = 1.0
    w_exc_cpg: float = 0.25
    w_cpg_inh: float = 10.0
    w_inh_cpg: float = 10.0
    w_cross: float = 0.05
    w_max_factor: float = 4.0        # w_max = faktor * početna srednja težina
    detune_lo: float = 0.1
    detune_hi: float = 0.3
    # plastičnost
    eta: float = 0.02                # 1/s po jedinici DA
    tau_c_ms: float = 1000.0
    tau_stdp_ms: float = 20.0
    A_plus: float = 1.0
    A_minus: float = 1.05
    # dopamin
    da_tonic: float = 0.1
    da_gain: float = 1.0
    da_cap: float = 10.0
    tau_stat_ms: float = 2000.0
    tau_habit_ms: float = 10000.0
    novelty_eps: float = 0.05
    # izlaz
    tau_rate_ms: float = 50.0
    out_gain_nA_per_Hz: float = 0.05

    def to_dict(self):
        return asdict(self)


def noise_scale(dt_ms: float) -> float:
    """Skala šuma po koraku: sqrt(0.1/dt). ASSUMPTION: noise_nA je definisan pri referentnom
    dt = 0.1 ms; pri tom koraku skala je tačno 1.0 (rezultat bit-identičan ranijem)."""
    dt = float(dt_ms)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt mora biti konačan i > 0, dobijeno {dt_ms!r}")
    return math.sqrt(0.1 / dt)


class Organism:
    def __init__(self, config: OrganismConfig | None = None, seed: int = 0,
                 plasticity: bool = True, dopamine: bool = True):
        self.cfg = c = config or OrganismConfig()
        self.plasticity = bool(plasticity)
        self.dopamine = bool(dopamine)
        rng = np.random.default_rng(seed)
        self._rng = rng
        ns = N_SENS * PER_SENS
        ne, ni, m, k = c.n_exc, c.n_inh, c.cpg_pop_size, c.cpg_inh_size
        self.sl_s = slice(0, ns)
        self.sl_e = slice(ns, ns + ne)
        self.sl_i = slice(ns + ne, ns + ne + ni)
        o = ns + ne + ni
        self.sl_c = slice(o, o + 4 * m)
        self.sl_ci = slice(o + 4 * m, o + 4 * m + 4 * k)
        N = o + 4 * m + 4 * k
        self.N = N

        def rnd(npost, npre, p):
            return (rng.random((npost, npre)) < p).astype(float)

        We = np.zeros((N, N))   # [post, pre], AMPA, nS
        Wi = np.zeros((N, N))   # GABA, nS
        plastic = np.zeros((N, N), bool)

        def put(W, post, pre, mask, w):
            W[post, pre] = mask * w * rng.uniform(0.5, 1.5, mask.shape)

        e, i_, s, cc, ci = self.sl_e, self.sl_i, self.sl_s, self.sl_c, self.sl_ci
        mask = rnd(ne, ns, 0.2)
        put(We, e, s, mask, c.w_sens_exc)
        plastic[e, s] = mask > 0
        put(We, i_, s, rnd(ni, ns, c.p_conn), c.w_sens_exc)
        put(We, e, e, rnd(ne, ne, c.p_conn), c.w_ee)
        put(We, i_, e, rnd(ni, ne, c.p_conn), c.w_ei)
        put(Wi, e, i_, rnd(ne, ni, c.p_conn), c.w_ie)
        put(Wi, i_, i_, rnd(ni, ni, c.p_conn), c.w_ii)
        mask = rnd(4 * m, ne, c.p_conn)
        put(We, cc, e, mask, c.w_exc_cpg)
        plastic[cc, e] = mask > 0
        # CPG: populacija g (A1,A2,B1,B2) pobuđuje svoje interneurone, oni inhibiraju partnera
        base_c, base_ci = cc.start, ci.start
        partner = {0: 1, 1: 0, 2: 3, 3: 2}
        for g in range(4):
            pe = slice(base_c + g * m, base_c + (g + 1) * m)
            pi = slice(base_ci + g * k, base_ci + (g + 1) * k)
            qe = slice(base_c + partner[g] * m, base_c + (partner[g] + 1) * m)
            put(We, pi, pe, rnd(k, m, 0.6), c.w_cpg_inh)
            put(Wi, qe, pi, rnd(m, k, 0.8), c.w_inh_cpg)
        # unakrsne A<->B veze: slabe, nasumične, plastične
        for ga in (0, 1):
            for gb in (2, 3):
                pa = slice(base_c + ga * m, base_c + (ga + 1) * m)
                pb = slice(base_c + gb * m, base_c + (gb + 1) * m)
                for post, pre in ((pb, pa), (pa, pb)):
                    mask = rnd(m, m, 0.3)
                    put(We, post, pre, mask, c.w_cross)
                    plastic[post, pre] = mask > 0
        # autapse: dijagonale se nuluju posle izvlačenja (RNG redosled isti); plastični blokovi
        # ne sadrže dijagonalu pune matrice, pa je STDP ne može vratiti
        np.fill_diagonal(We, 0.0)
        np.fill_diagonal(Wi, 0.0)
        self._WeT = np.ascontiguousarray(We.T)
        self._WiT = np.ascontiguousarray(Wi.T)
        cross_mask = np.zeros((4 * m, 4 * m))
        for ga in (0, 1):
            for gb in (2, 3):
                for g_post, g_pre in ((gb, ga), (ga, gb)):
                    cross_mask[g_post * m:(g_post + 1) * m, g_pre * m:(g_pre + 1) * m] = 1.0
        cross_mask *= plastic[cc, cc]
        # blokovi: ime -> (pre slice, post slice, maska [post, pre])
        self._blocks = {
            "sens_exc": (s, e, plastic[e, s].astype(float)),
            "exc_cpg": (e, cc, plastic[cc, e].astype(float)),
            "cross": (cc, cc, cross_mask),
        }
        w0_all = np.concatenate([We[po, pr][mk > 0] for pr, po, mk in self._blocks.values()])
        self._wmax = c.w_max_factor * float(w0_all.mean())
        g_da = c.eta * 1e-3
        self._g_da = g_da
        self.rules, self._rs = {}, {}
        for key, (pr, po, mk) in self._blocks.items():
            rule = DASTDP(n_pre=pr.stop - pr.start, n_post=po.stop - po.start, name=key,
                          tau_c=c.tau_c_ms, tau_d=TAU_D_MS, tau_plus=c.tau_stdp_ms,
                          tau_minus=c.tau_stdp_ms, A_plus=c.A_plus, A_minus=c.A_minus,
                          w_min=0.0, w_max=1.0, d_tonic=(g_da * c.da_tonic) if self.dopamine else 0.0)
            st = rule.init_state(None, rng)
            st["w"] = We[po, pr] / self._wmax * mk
            self.rules[key], self._rs[key] = rule, st

        self.V = c.E_L + rng.uniform(0, 8, N)
        self._V0 = self.V.copy()
        self._ref = np.zeros(N)
        self._wad = np.zeros(N)
        tau_w = np.full(N, c.tau_w_ms)
        b = np.zeros(N)
        self.detune = float(rng.uniform(c.detune_lo, c.detune_hi))
        factors = np.array([1.0, 1.0, 1 + self.detune, 1 + self.detune])
        tau_w[cc] = np.repeat(c.tau_w_ms * factors, m)
        b[cc] = c.w_adapt_inc_pA
        self._tau_w, self._b = tau_w, b
        bias = np.zeros(N)
        bias[s] = c.sensory_bias_nA
        bias[e] = c.bias_exc_nA
        bias[i_] = c.bias_inh_nA
        bias[cc] = c.bias_cpg_nA
        self._bias = bias
        pol = np.ones((N_SENS, PER_SENS))
        pol[:, 3:] = -1.0   # on/off aferenti
        self._pol = pol.ravel()
        self.ge = np.zeros(N)
        self.gi = np.zeros(N)
        self._rate = np.zeros(4)
        self._mu = np.zeros(N_SENS)
        self._var = np.full(N_SENS, c.novelty_eps ** 2)
        self._nbar = 0.0
        self._first = True
        self.da_phasic = 0.0
        self._k = 0
        self._noise_i = 512
        self._noise: np.ndarray | None = None
        self._dt = None
        self.last_spikes = np.zeros(N, bool)

    def _consts(self, dt):
        c = self.cfg
        self._dt = dt
        self._noise_k = noise_scale(dt)
        self._d_ampa = np.exp(-dt / c.tau_ampa_ms)
        self._d_gaba = np.exp(-dt / c.tau_gaba_ms)
        self._d_w = np.exp(-dt / self._tau_w)
        self._d_rate = np.exp(-dt / c.tau_rate_ms)
        self._a_stat = dt / c.tau_stat_ms
        self._a_hab = dt / c.tau_habit_ms

    def weight_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Kopije punih matrica (We, Wi) u obliku [post, pre], nS."""
        return self._WeT.T.copy(), self._WiT.T.copy()

    def reset(self) -> None:
        """Vraća dinamičko stanje na početno. Zadržava naučene težine w, _WeT/_WiT i detune.
        Bafer šuma i _noise_i se namerno ne resetuju (ASSUMPTION: šum se ne reseeduje)."""
        c = self.cfg
        self.V = self._V0.copy()
        self._ref = np.zeros(self.N)
        self._wad = np.zeros(self.N)
        self.ge = np.zeros(self.N)
        self.gi = np.zeros(self.N)
        self._rate = np.zeros(4)
        self._mu = np.zeros(N_SENS)
        self._var = np.full(N_SENS, c.novelty_eps ** 2)
        self._nbar = 0.0
        self._first = True
        self.da_phasic = 0.0
        self.last_spikes = np.zeros(self.N, bool)
        self._k = 0
        for key, rule in self.rules.items():
            st = self._rs[key]
            st["c"] = np.zeros_like(st["c"])
            st["x_pre"] = np.zeros_like(st["x_pre"])
            st["x_post"] = np.zeros_like(st["x_post"])
            st["d"] = np.full(1, rule.p["d_tonic"])

    def plastic_weights(self) -> np.ndarray:
        """Maskirani plastični ulazi (sens_exc, exc_cpg, cross), nS."""
        return np.concatenate([(self._rs[k]["w"] * self._wmax)[self._blocks[k][2] > 0] for k in self._blocks])

    def cross_weights(self) -> np.ndarray:
        return (self._rs["cross"]["w"] * self._wmax)[self._blocks["cross"][2] > 0]

    def _plasticity_step(self, t, dt, spk):
        c = self.cfg
        da = self._g_da * self.da_phasic * dt / TAU_D_MS
        for key, rule in self.rules.items():
            pr, po, mk = self._blocks[key]
            st, _ = rule.step(t, dt, self._rs[key], {"pre_spikes": spk[pr], "post_spikes": spk[po], "da": da})
            st["w"] *= mk
            st["c"] *= mk
            self._rs[key] = st
            self._WeT[pr, po] = (st["w"] * self._wmax).T

    def _novelty(self, s):
        c = self.cfg
        if self._first:
            self._mu = s.copy()
            self._first = False
        d = s - self._mu
        n = float((np.abs(d) / (np.sqrt(self._var) + c.novelty_eps)).sum() / N_SENS)
        self._mu = self._mu + self._a_stat * d
        self._var = self._var + self._a_stat * (d * d - self._var)
        phasic = c.da_gain * max(0.0, n - self._nbar)
        self._nbar += self._a_hab * (n - self._nbar)
        return min(phasic, c.da_cap)

    def step(self, sensory_current_nA: np.ndarray, dt_ms: float) -> np.ndarray:
        c = self.cfg
        dt = float(dt_ms)
        if dt != self._dt:
            self._consts(dt)
        s = np.asarray(sensory_current_nA, float)
        if s.size != N_SENS:
            raise ValueError(f"očekivano 9 aferentnih struja, dobijeno {s.size}")
        s = s.reshape(N_SENS)
        if not np.isfinite(s.sum()):
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        self.da_phasic = self._novelty(s) if self.dopamine else 0.0

        if self._noise is None or self._noise_i >= 512:
            self._noise = self._rng.standard_normal((512, self.N)) * c.noise_nA
            self._noise_i = 0
        I = self._bias + self._noise[self._noise_i] * self._noise_k
        self._noise_i += 1
        I[self.sl_s] += c.sensory_gain * self._pol * np.repeat(s, PER_SENS)
        V = self.V
        dV = (-c.gL_nS * (V - c.E_L) - self.ge * (V - c.E_ampa) - self.gi * (V - c.E_gaba)
              + 1000.0 * I - self._wad) / c.C_pF
        V = V + dt * dV
        V = np.where(self._ref > 0, c.V_reset, V)
        self._ref -= dt
        spk = V >= c.V_th
        V[spk] = c.V_reset
        self._ref[spk] = c.t_ref_ms
        self.V = V
        self._wad = self._wad * self._d_w + self._b * spk
        self.ge *= self._d_ampa
        self.gi *= self._d_gaba
        idx = np.flatnonzero(spk)
        if idx.size:
            self.ge += self._WeT[idx].sum(0)
            self.gi += self._WiT[idx].sum(0)
            cnt = spk[self.sl_c].reshape(4, -1).sum(1)
            self._rate = self._rate * self._d_rate + cnt / c.cpg_pop_size * (1000.0 / c.tau_rate_ms)
        else:
            self._rate = self._rate * self._d_rate
        self.last_spikes = spk
        self._k += 1
        if self.plasticity:
            self._plasticity_step(self._k * dt, dt, spk)
        return np.maximum(0.0, c.out_gain_nA_per_Hz * self._rate)
