# Specifikacije mehanizama

Sklop: [hh](hh.md), [synapses](synapses.md), [stdp](stdp.md), [da_stdp](da_stdp.md), [homeostasis](homeostasis.md), [cpg](cpg.md)

Tabela pragova. Svaki prag je provera kvalitativne verodostojnosti, ne poređenje sa eksperimentalnim podacima. Svaka stavka iz odeljka „## 10.“ ima po jedan test u `tests/mechanisms/test_spec_thresholds.py` (izmerene vrednosti se ispisuju sa `pytest -s`). Ugovor: `tests/contract/test_contract_specs_bibliography.py`.

| mehanizam | prag | ugovorni test |
|---|---|---|
| hh | mirovanje −65 ± 0.5 mV | `tests/mechanisms/test_spec_thresholds.py::test_hh_rest_minus65` |
| hh | vrh > +20 mV | `tests/mechanisms/test_spec_thresholds.py::test_hh_peak_above_20mV` |
| hh | amplituda > 80 mV od mirovanja | `tests/mechanisms/test_spec_thresholds.py::test_hh_amplitude_above_80mV` |
| hh | poluširina 0.5–2.5 ms (pulsni stimulus 40 µA/cm², 1 ms) | `tests/mechanisms/test_spec_thresholds.py::test_hh_half_width_0p5_to_2p5_ms` |
| hh | I = 1 µA/cm² ne daje spajk | `tests/mechanisms/test_spec_thresholds.py::test_hh_I1_no_spike` |
| hh | I = 12 µA/cm² daje ≥ 8 spajkova u 200 ms | `tests/mechanisms/test_spec_thresholds.py::test_hh_I12_at_least_8_spikes_in_200ms` |
| hh | f-I za I ∈ {12, 20, 35, 50} neopadajuća | `tests/mechanisms/test_spec_thresholds.py::test_hh_fI_nondecreasing` |
| hh | dt ∈ {0.01, 0.025, 0.05}: broj spajkova ±1, prvi spajk ±0.5 ms, vrh ±3 mV naspram dt = 0.01 | `tests/mechanisms/test_spec_thresholds.py::test_hh_dt_consistency` |
| synapses | vrh = g_max ± 2% | `tests/mechanisms/test_spec_thresholds.py::test_synapse_peak_equals_g_max` |
| synapses | I(E_rev) = 0 | `tests/mechanisms/test_spec_thresholds.py::test_synapse_current_zero_at_reversal` |
| synapses | B(−65) prema formuli | `tests/mechanisms/test_spec_thresholds.py::test_synapse_nmda_mg_block_formula_at_minus65` |
| synapses | NMDA sporije od GABA_A, a GABA_A sporije od AMPA | `tests/mechanisms/test_spec_thresholds.py::test_synapse_kinetics_order_nmda_gaba_ampa` |
| stdp | znak Δw prema Δt | `tests/mechanisms/test_spec_thresholds.py::test_stdp_sign_of_dw` |
| stdp | fit τ ± 10% | `tests/mechanisms/test_spec_thresholds.py::test_stdp_tau_fit_within_10_percent` |
| stdp | online pravilo = analitičko | `tests/mechanisms/test_spec_thresholds.py::test_stdp_online_equals_analytic` |
| da_stdp | bez DA Δw = 0 tačno | `tests/mechanisms/test_spec_thresholds.py::test_da_stdp_no_da_no_change` |
| da_stdp | c(τc)/c0 = e⁻¹ ± 2% | `tests/mechanisms/test_spec_thresholds.py::test_da_stdp_eligibility_decays_to_1_over_e` |
| da_stdp | nagrada posle pre→post daje Δw > 0, posle post→pre Δw < 0 | `tests/mechanisms/test_spec_thresholds.py::test_da_stdp_reward_sign_follows_order` |
| da_stdp | nagrada odložena 5 s daje < 10% efekta nagrade od 0.89 s | `tests/mechanisms/test_spec_thresholds.py::test_da_stdp_delay_5s_less_than_10_percent_of_0p89s` |
| homeostasis | frekvencija konvergira ka cilju ± 5% odozdo i odozgo | `tests/mechanisms/test_spec_thresholds.py::test_homeostasis_rate_converges_to_target_within_5_percent` |
| homeostasis | odnosi težina sačuvani do 1e-6 | `tests/mechanisms/test_spec_thresholds.py::test_homeostasis_weight_ratios_preserved_to_1e6` |
| cpg | korelacija y1 i y2 < −0.3 | `tests/mechanisms/test_spec_thresholds.py::test_cpg_anticorrelation_below_minus_0p3` |
| cpg | istovremena aktivnost < 20% vremena | `tests/mechanisms/test_spec_thresholds.py::test_cpg_coactivity_below_20_percent` |
| cpg | ≥ 3 ciklusa u 5 s | `tests/mechanisms/test_spec_thresholds.py::test_cpg_at_least_3_cycles_in_5s` |
| cpg | kontrola a = 0 nema ritam (std < 1e-3) | `tests/mechanisms/test_spec_thresholds.py::test_cpg_control_a0_no_rhythm_std_below_1e3` |
