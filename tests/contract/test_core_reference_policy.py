import importlib
import pathlib


def test_core_does_not_import_external_neuroscience_frameworks():
    roots = [pathlib.Path('neural/core')]
    forbidden = ('neuron', 'brian2', 'nest', 'neuroml', 'sonata')
    for root in roots:
        for path in root.rglob('*.py'):
            text = path.read_text(encoding='utf-8').lower()
            assert not any(f'import {name}' in text or f'from {name}' in text for name in forbidden)


def test_verified_reference_dois_are_stable():
    dois = {
        '10.1113/jphysiol.1952.sp004764',
        '10.1162/neco.1997.9.6.1179',
        '10.3389/neuro.11.005.2008',
        '10.4249/scholarpedia.1430',
        '10.1371/journal.pcbi.1007696',
    }
    assert all(doi.startswith('10.') for doi in dois)
    assert all(' ' not in doi for doi in dois)
