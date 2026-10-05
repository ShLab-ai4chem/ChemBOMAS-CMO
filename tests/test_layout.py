from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_task_inputs_exist():
    expected = [
        ROOT / "data/wet/00-basic/FD_wqp/search_space.csv",
        ROOT / "data/wet/01-cluster/FD_wqp/order.json",
        ROOT / "data/dry/00-basic/EDBO/search_space.csv",
        ROOT / "data/dry/01-cluster/EDBO/partition_expert.json",
    ]
    assert all(path.exists() for path in expected)


def test_bo_package_imports():
    from chembomas.bo_core.data.bodata import BoData
    from chembomas.bo_core.mcts.tree import MCTS

    assert BoData is not None
    assert MCTS is not None
