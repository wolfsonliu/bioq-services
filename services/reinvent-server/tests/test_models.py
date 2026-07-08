import pytest
from pydantic import ValidationError


def test_sampling_defaults():
    from server.models import SamplingRequest
    r = SamplingRequest()
    assert r.generator == "reinvent"
    assert r.num_smiles == 100
    assert r.sample_strategy == "multinomial"
    assert r.model_file is None


def test_scoring_requires_scoring_dict():
    from server.models import ScoringRequest
    with pytest.raises(ValidationError):
        ScoringRequest()  # scoring is required
    r = ScoringRequest(scoring={"type": "geometric_mean", "component": []})
    assert r.smiles_column == "SMILES"


def test_staged_learning_stage_spec():
    from server.models import StagedLearningRequest, StageSpec
    r = StagedLearningRequest(
        stages=[StageSpec(chkpt_name="s1.chkpt",
                          scoring={"type": "geometric_mean", "component": []})],
    )
    assert r.learning_strategy["type"] == "dap"
    assert r.stages[0].max_steps == 100
    assert r.generator == "reinvent"


def test_transfer_learning_defaults():
    from server.models import TransferLearningRequest
    r = TransferLearningRequest()
    assert r.output_model_name == "TL_model.model"
    assert r.num_epochs == 3
    assert r.pairs is None
