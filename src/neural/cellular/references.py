"""Source records used by the cellular and synaptic modules.

doi_status 'verified' is only used for the DOIs confirmed by the literature
agent (ag-research); all others stay 'needs-verify' or 'TBD-verify'.
Numeric values in the code are either taken from the cited paper or are
numerical test fixtures; each module documents which.
"""


def _ref(source_id, title, authors, year, doi, doi_status):
    return {'source_id': source_id, 'title': title, 'authors': authors,
            'year': year, 'doi': doi, 'doi_status': doi_status}


HH1952 = _ref('HH1952', 'A quantitative description of membrane current and its application '
              'to conduction and excitation in nerve', 'Hodgkin AL, Huxley AF', 1952,
              '10.1113/jphysiol.1952.sp004764', 'verified')
HH1952_CURRENTS = _ref('HH1952_currents', 'Currents carried by sodium and potassium ions through '
                       'the membrane of the giant axon of Loligo', 'Hodgkin AL, Huxley AF', 1952,
                       '10.1113/jphysiol.1952.sp004717', 'verified')
HH1952_CONDUCTANCE = _ref('HH1952_conductance', 'The components of membrane conductance in the '
                          'giant axon of Loligo', 'Hodgkin AL, Huxley AF', 1952,
                          '10.1113/jphysiol.1952.sp004718', 'verified')
RALL1959 = _ref('Rall1959', 'Branching dendritic trees and motoneuron membrane resistivity',
                'Rall W', 1959, '10.1016/0014-4886(59)90046-9', 'needs-verify')
HINES1984 = _ref('Hines1984', 'Efficient computation of branching nerve equations', 'Hines M',
                 1984, 'TBD-verify', 'TBD-verify')
JAHR_STEVENS1990 = _ref('JahrStevens1990', 'Voltage dependence of NMDA-activated macroscopic '
                        'conductances predicted by single-channel kinetics',
                        'Jahr CE, Stevens CF', 1990, '10.1523/JNEUROSCI.10-09-03178.1990',
                        'needs-verify')
DAYAN_ABBOTT2001 = _ref('DayanAbbott2001', 'Theoretical Neuroscience', 'Dayan P, Abbott LF', 2001,
                        'TBD-verify', 'TBD-verify')
TM1997 = _ref('TM1997', 'The neural code between neocortical pyramidal neurons depends on '
              'neurotransmitter release probability', 'Tsodyks MV, Markram H', 1997,
              '10.1073/pnas.94.2.719', 'needs-verify')
MARKRAM1998 = _ref('Markram1998', 'Differential signaling via the same axon of neocortical '
                   'pyramidal neurons', 'Markram H, Wang Y, Tsodyks M', 1998,
                   '10.1073/pnas.95.9.5323', 'needs-verify')
BIPOO1998 = _ref('BiPoo1998', 'Synaptic modifications in cultured hippocampal neurons',
                 'Bi GQ, Poo MM', 1998, '10.1523/JNEUROSCI.18-24-10464.1998', 'needs-verify')
BIPOO2001 = _ref('BiPoo2001', 'Synaptic modification by correlated activity: Hebb\'s postulate '
                 'revisited', 'Bi G, Poo M', 2001, '10.1146/annurev.neuro.24.1.139',
                 'needs-verify')
SONG2000 = _ref('Song2000', 'Competitive Hebbian learning through spike-timing-dependent '
                'synaptic plasticity', 'Song S, Miller KD, Abbott LF', 2000,
                '10.1038/78829', 'needs-verify')
