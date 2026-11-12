import pathlib


def test_cell_modules_do_not_depend_on_external_simulators():
    forbidden = ('import neuron', 'from neuron', 'import brian2', 'from brian2', 'import nest', 'from nest')
    for root_name in ('neural/cell', 'neural/synapse'):
        root = pathlib.Path(root_name)
        for path in root.rglob('*.py'):
            text = path.read_text(encoding='utf-8').lower()
            assert not any(token in text for token in forbidden)


def test_cell_code_avoids_nonfinite_literals_in_source():
    for root_name in ('neural/cell', 'neural/synapse'):
        root = pathlib.Path(root_name)
        for path in root.rglob('*.py'):
            text = path.read_text(encoding='utf-8').lower()
            assert 'nan' not in text and 'inf' not in text
