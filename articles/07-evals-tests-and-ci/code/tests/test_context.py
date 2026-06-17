"""Unit tests for the human-in-the-loop approval logic in the run context."""

from data_agent.context import PipelineContext


def test_auto_approve_allows_without_asking():
    asked = []
    ctx = PipelineContext(auto_approve=True, approver=lambda a, d: asked.append(a) or False)
    assert ctx.approve("run_python_file", "x") is True
    assert asked == []  # approver never consulted when auto-approving


def test_no_approver_defaults_to_allow():
    # Non-interactive (e.g. tests / automation): nothing to ask, so allow.
    ctx = PipelineContext()
    assert ctx.approve("write_file", "x") is True


def test_approver_decision_is_honored():
    ctx_yes = PipelineContext(approver=lambda a, d: True)
    ctx_no = PipelineContext(approver=lambda a, d: False)
    assert ctx_yes.approve("run_python_file", "x") is True
    assert ctx_no.approve("run_python_file", "x") is False


def test_approver_receives_action_and_detail():
    seen = {}

    def approver(action, detail):
        seen["action"], seen["detail"] = action, detail
        return True

    PipelineContext(approver=approver).approve("write_file", "path: clean.py")
    assert seen == {"action": "write_file", "detail": "path: clean.py"}
