"""Tier 1: the typed constraint models — parsing, the registry, hashability, the consume set the schedule reads, the start policy, bounds, and every residual."""

import json
from dataclasses import replace
from decimal import Decimal
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.domain.constraints import (
    CashLimit,
    ColumnRef,
    ConstraintSpecError,
    ExposureLimit,
    GroupLimit,
    ParticipationLimit,
    ScalarRef,
    TurnoverLimit,
    TypedConstraint,
    WeightLimit,
    check_against_spec,
    constraint_kinds,
    consumed_securities,
    effective_bounds,
    frame_reads_chain,
    parse_constraint,
    parse_constraints,
    register_constraint_kind,
    starting_values,
    vector_values,
)
from portfolio_optimizer.domain.order_flow import INFLOW, OUTFLOW
from portfolio_optimizer.domain.registry import kind_name
from portfolio_optimizer.domain.results import ChainState, Contribution, ProblemSpec
from tests.conftest import Factories


def rows(*records: dict[str, object]) -> pd.DataFrame:
    """A constraints frame the way the loader would deliver it: params as JSON text, kind as a column."""
    filled = [{"portfolio_id": "P1", **record} for record in records]
    frame = pd.DataFrame.from_records(filled)
    if "params" in frame.columns:
        frame["params"] = frame["params"].map(lambda value: json.dumps(value) if isinstance(value, dict) else value)
    return frame


def flagged_spec(make: Factories, **overrides: object) -> ProblemSpec:
    """The canonical three-security spec with an ``is_thin`` flag on S2 alone."""
    return make.spec(flags={"is_thin": np.array([False, False, True])}, **overrides)


# --- parsing, the registry, and hashability ---


def test_frames_that_do_not_speak_the_spec_parse_to_none() -> None:
    assert parse_constraints(pd.DataFrame()) is None
    assert parse_constraints(rows({"name": "long_only"})) is None, "no kind column: a different vocabulary, untouched, for a custom solve step to read"


def test_rows_parse_into_their_models_and_the_label_column_names_them() -> None:
    parsed = parse_constraints(
        rows({"kind": "weight_limit", "label": "cap", "params": {"direction": "<=", "bounds": "0.1"}}, {"kind": "cash_limit", "name": "floor", "params": {"direction": ">=", "bounds": "0"}})
    )
    assert parsed is not None
    assert parsed.typed == (WeightLimit(name="cap", direction="<=", bounds=Decimal("0.1")), CashLimit(name="floor", direction=">=", bounds=Decimal(0)))
    assert not parsed.reads_chain


@pytest.mark.parametrize(
    ("record", "fragment"),
    [
        ({"kind": "weight_limit", "label": "cap", "params": {"direction": "sideways", "bounds": "0.1"}}, r"constraints\[0\]: weight_limit: direction"),
        ({"kind": None, "label": "cap", "params": {"direction": "<=", "bounds": "0.1"}}, r"constraints\[0\]: the row names no kind; known kinds: \["),
        ({"kind": "no_such_kind", "label": "cap"}, r"constraints\[0\]: unknown kind 'no_such_kind'; known kinds: \["),
        ({"kind": "weight_limit", "label": "cap", "params": {"direction": "<=", "bounds": "0.1", "colour": "red"}}, r"constraints\[0\]: weight_limit: colour: Extra inputs are not permitted"),
    ],
    ids=["bad direction", "no kind", "unknown kind", "unknown field"],
)
def test_a_malformed_row_names_itself(record: dict[str, object], fragment: str) -> None:
    with pytest.raises(ConstraintSpecError, match=fragment):
        parse_constraints(rows(record))


def test_duplicate_names_are_refused() -> None:
    duplicated = rows({"kind": "weight_limit", "label": "cap", "params": {"direction": "<=", "bounds": "0.1"}}, {"kind": "weight_limit", "label": "cap", "params": {"direction": ">=", "bounds": "0"}})
    with pytest.raises(ConstraintSpecError, match="'cap' is also used"):
        parse_constraints(duplicated)


def test_typed_constraints_are_hashable_and_bounds_dictionaries_canonicalize() -> None:
    one = GroupLimit(name="bands", direction="<=", column="sector", bounds={"TECH": Decimal("0.5"), "HEALTH": Decimal("0.3")})
    other = GroupLimit(name="bands", direction="<=", column="sector", bounds={"HEALTH": "0.3", "TECH": "0.5"})
    assert one == other
    assert len({one, other, ParticipationLimit(name="adv", direction="<=")}) == 2


def test_a_record_round_trips_through_the_registry() -> None:
    model = GroupLimit(name="bands", direction="<=", column="sector", bounds={"TECH": Decimal("0.5")}, scope="is_thin", tolerance=Decimal("0.01"))
    record = model.record()
    assert record["kind"] == "group_limit" and record["bounds"] == [["TECH", "0.5"]]
    assert parse_constraint(record) == model
    assert kind_name(GroupLimit) == "group_limit"


def test_the_registry_holds_the_shipped_kinds_and_accepts_a_registered_one() -> None:
    assert set(constraint_kinds()) >= {"weight_limit", "group_limit", "exposure_limit", "cash_limit", "turnover_limit", "participation_limit"}

    @register_constraint_kind
    class NoNewNames(TypedConstraint):
        """A kind a desk might publish."""

        kind: Literal["no_new_names"] = "no_new_names"

    assert constraint_kinds()["no_new_names"] is NoNewNames
    parsed = parse_constraint({"kind": "no_new_names", "name": "freeze", "direction": "<="})
    assert isinstance(parsed, NoNewNames)
    with pytest.raises(ConstraintSpecError, match="test: unknown kind 'nothing'"):
        parse_constraint({"kind": "nothing", "name": "x", "direction": "<="}, where="test")


# --- the consume set: what the schedule reads ---


def test_nothing_that_reads_the_chain_means_no_consume_set(make: Factories) -> None:
    spec = flagged_spec(make)
    parsed = parse_constraints(rows({"kind": "weight_limit", "label": "cap", "params": {"direction": "<=", "bounds": "0.9"}}))
    assert consumed_securities(parsed, spec, INFLOW, chain_aware_terms=False, opaque_solve=False) == ()
    assert consumed_securities(None, spec, INFLOW, chain_aware_terms=False, opaque_solve=False) == (), "no constraints at all reads nothing"


def test_a_scoped_chain_constraint_consumes_its_scope_of_the_tradable_set(make: Factories) -> None:
    spec = flagged_spec(make)
    parsed = parse_constraints(rows({"kind": "participation_limit", "label": "adv", "params": {"direction": "<=", "scope": "is_thin"}}))
    assert consumed_securities(parsed, spec, INFLOW, chain_aware_terms=False, opaque_solve=False) == ("S2",)
    unscoped = parse_constraints(rows({"kind": "participation_limit", "label": "adv", "params": {"direction": "<="}}))
    assert consumed_securities(unscoped, spec, INFLOW, chain_aware_terms=False, opaque_solve=False) == ("S0", "S1", "S2")


@pytest.mark.parametrize(("chain_aware_terms", "opaque_solve"), [(True, False), (False, True)], ids=["chain-term", "opaque-solve-step"])
def test_anything_opaque_widens_the_consume_set_to_the_whole_tradable_set(make: Factories, chain_aware_terms: bool, opaque_solve: bool) -> None:
    spec = flagged_spec(make)
    consumed = consumed_securities(None, spec, INFLOW, chain_aware_terms=chain_aware_terms, opaque_solve=opaque_solve)
    assert consumed == ("S0", "S1", "S2"), "an opaque reader cannot declare a narrower scope"


def test_the_consume_set_respects_the_side_profiles_tradable_set(make: Factories) -> None:
    spec = flagged_spec(make, w0=np.array([0.5, 0.5, 0.0]))
    parsed = parse_constraints(rows({"kind": "participation_limit", "label": "adv", "params": {"direction": "<="}}))
    assert consumed_securities(parsed, spec, OUTFLOW, chain_aware_terms=False, opaque_solve=False) == ("S0", "S1"), "an outflow couples through what is sellable, and S2 is not held"


def test_a_whole_books_rows_say_whether_anything_reads_the_chain() -> None:
    assert not frame_reads_chain(pd.DataFrame())
    assert not frame_reads_chain(rows({"kind": "cash_limit", "label": "floor", "params": {"direction": ">=", "bounds": "0"}}))
    assert frame_reads_chain(
        rows({"kind": "cash_limit", "label": "floor", "params": {"direction": ">=", "bounds": "0"}}, {"kind": "participation_limit", "label": "adv", "params": {"direction": "<="}})
    )
    assert frame_reads_chain(rows({"name": "long_only"})), "another vocabulary might read anything"
    assert frame_reads_chain(rows({"kind": None, "label": "x"})), "a row without a kind might read anything"


# --- the start policy and the shared bound arithmetic ---


def test_effective_bounds_loosen_to_the_current_value_only_when_allowed() -> None:
    bounds = np.array([0.4, 0.4])
    current = np.array([0.5, 0.3])
    np.testing.assert_allclose(effective_bounds("<=", False, bounds, current), [0.4, 0.4])
    np.testing.assert_allclose(effective_bounds("<=", True, bounds, current), [0.5, 0.4], err_msg="a breached start holds; a clean one keeps the bound")
    np.testing.assert_allclose(effective_bounds(">=", True, np.array([0.2, 0.2]), np.array([0.1, 0.3])), [0.1, 0.2])
    np.testing.assert_allclose(effective_bounds("<", True, bounds, current), effective_bounds("<=", True, bounds, current), err_msg="the strict spelling binds identically")


def test_participation_limit_refuses_shapes_that_mean_nothing() -> None:
    with pytest.raises(ValueError, match="only bounds from above"):
        ParticipationLimit(name="adv", direction=">=")
    with pytest.raises(ValueError, match="allow_current_weight does not apply"):
        ParticipationLimit(name="adv", direction="<=", allow_current_weight=True)
    with pytest.raises(ValueError, match="cash_limit takes no scope"):
        CashLimit(name="cash", direction="<=", bounds=Decimal(1), scope="is_thin")


# --- residuals: the verifier's half of the contract ---


def test_weight_limit_residual_checks_only_the_scope_and_honours_the_start_policy(make: Factories) -> None:
    spec = flagged_spec(make, w0=np.array([0.5, 0.3, 0.2]))
    chain = ChainState.empty(spec.security_ids)
    solution = make.solution(spec, w=np.array([0.5, 0.3, 0.2]))
    strict = WeightLimit(name="cap", direction="<=", bounds=Decimal("0.4"))
    ((_, residual),) = strict.residual(spec, solution, chain, INFLOW)
    np.testing.assert_allclose(residual, [0.1, -0.1, -0.2], err_msg="S0 breaches the cap it started above")
    held = WeightLimit(name="cap", direction="<=", bounds=Decimal("0.4"), allow_current_weight=True)
    ((_, residual),) = held.residual(spec, solution, chain, INFLOW)
    np.testing.assert_allclose(residual, [0.0, -0.1, -0.2], err_msg="the breached start loosens its own bound to the holding, exactly")
    scoped = WeightLimit(name="cap", direction="<=", bounds=Decimal("0.1"), scope="is_thin")
    ((_, residual),) = scoped.residual(spec, solution, chain, INFLOW)
    np.testing.assert_allclose(residual, [0.0, 0.0, 0.1], err_msg="outside the scope the residual is zero by construction")


def test_bounds_may_come_from_a_spec_column_or_an_account_scalar(make: Factories) -> None:
    spec = make.spec(scalars={"max_single": 0.4}, columns={"cap": np.array([0.2, 0.5, 0.5])})
    chain = ChainState.empty(spec.security_ids)
    solution = make.solution(spec, w=np.array([0.5, 0.3, 0.2]))
    from_scalar = WeightLimit(name="cap", direction="<=", bounds=ScalarRef(scalar="max_single"))
    ((_, residual),) = from_scalar.residual(spec, solution, chain, INFLOW)
    np.testing.assert_allclose(residual, [0.1, -0.1, -0.2])
    from_column = WeightLimit(name="cap", direction="<=", bounds=ColumnRef(column="cap"))
    ((_, residual),) = from_column.residual(spec, solution, chain, INFLOW)
    np.testing.assert_allclose(residual, [0.3, -0.2, -0.3])
    box = WeightLimit(name="floor", direction=">=", bounds=ColumnRef(column="lb"))
    ((_, residual),) = box.residual(spec, solution, chain, INFLOW)
    np.testing.assert_allclose(residual, [-0.5, -0.3, -0.2], err_msg="the spec's own vectors are columns too")


def test_group_limit_residual_bounds_named_groups_and_refuses_unknown_ones(make: Factories) -> None:
    spec = make.spec()  # one sector, TECH, holding everything
    chain = ChainState.empty(spec.security_ids)
    solution = make.solution(spec)
    bands = GroupLimit(name="bands", direction="<=", column="sector", bounds={"TECH": Decimal("0.8")})
    ((_, residual),) = bands.residual(spec, solution, chain, INFLOW)
    assert residual[0] == pytest.approx(1.0 - 0.8)
    with pytest.raises(ConstraintSpecError, match=r"group\(s\) \['ENERGY'\]"):
        GroupLimit(name="bands", direction="<=", column="sector", bounds={"ENERGY": Decimal("0.8")}).residual(spec, solution, chain, INFLOW)
    with pytest.raises(KeyError, match="spec has no grouping 'country'"):
        GroupLimit(name="bands", direction="<=", column="country", bounds=Decimal("0.5")).residual(spec, solution, chain, INFLOW)


def test_exposure_limit_residual_is_the_scoped_dot_product(make: Factories) -> None:
    spec = make.spec(columns={"beta": np.array([1.0, 2.0, 3.0])})
    solution = make.solution(spec)  # w0 = 1/3 each -> beta 2.0
    tight = ExposureLimit(name="beta_cap", direction="<=", column="beta", bounds=Decimal("1.5"))
    ((_, residual),) = tight.residual(spec, solution, ChainState.empty(spec.security_ids), INFLOW)
    assert residual[0] == pytest.approx(0.5)
    floor = ExposureLimit(name="beta_floor", direction=">", column="beta", bounds=Decimal("2.5"))  # the strict spelling: same closed bound
    ((_, residual),) = floor.residual(spec, solution, ChainState.empty(spec.security_ids), INFLOW)
    assert residual[0] == pytest.approx(0.5), "a ge bound is violated from below"


def test_cash_and_turnover_limits_read_the_accounts_scalars(make: Factories) -> None:
    spec = make.spec(cash_lb=0.05, cash_ub=0.1, max_turnover=0.2)
    chain = ChainState.empty(spec.security_ids)
    solution = make.solution(spec, w=np.array([0.3, 0.3, 0.3]), buy=np.array([0.0, 0.0, 0.0]), sell=np.array([1.0 / 3.0 - 0.3] * 3))
    floor = CashLimit(name="floor", direction=">=", bounds=ScalarRef(scalar="cash_lb"))
    ((_, residual),) = floor.residual(spec, solution, chain, INFLOW)
    assert residual[0] == pytest.approx(0.05 - 0.1), "cash after the run is 0.1, above the 0.05 floor"
    cap = CashLimit(name="cap", direction="<=", bounds=Decimal("0.05"))
    ((_, residual),) = cap.residual(spec, solution, chain, INFLOW)
    assert residual[0] == pytest.approx(0.05)
    turnover = TurnoverLimit(name="turnover", direction="<=", bounds=ScalarRef(scalar="max_turnover"))
    ((_, residual),) = turnover.residual(spec, solution, chain, INFLOW)
    assert residual[0] == pytest.approx(0.1 - 0.2), "a tenth of NAV was sold against a cap of a fifth"


def test_participation_residual_scopes_both_checks_and_reads_the_chain(make: Factories) -> None:
    spec = flagged_spec(make, adv_capacity=np.array([0.05, 0.05, 0.05]), price=np.full(3, 100.0))
    consumed = Contribution("P0", ("S2",), np.array([300.0]))  # 300 shares at 100 on NAV 1e6 = 0.03 of NAV
    chain = INFLOW.chain_state(spec, [consumed], np.ones(3, dtype=np.bool_))
    adv = ParticipationLimit(name="adv", direction="<=", scope="is_thin")
    solution = make.solution(spec, w=spec.w0 + np.array([0.0, 0.0, 0.04]), buy=np.array([0.0, 0.0, 0.04]))
    residuals = dict(adv.residual(spec, solution, chain, INFLOW))
    assert residuals["participation"][2] == pytest.approx(-0.01), "own budget has room in the scoped name; unscoped entries are zero by construction"
    assert residuals["cumulative_participation"][2] == pytest.approx(0.04 - 0.02), "the predecessor left 0.02 of S2's budget"
    assert residuals["cumulative_participation"][:2].max() == 0.0, "unscoped names are not checked"
    with pytest.raises(ConstraintSpecError, match="not aligned"):
        adv.remaining(spec, ChainState.empty(("S0",)))


def test_vector_values_and_starting_values_cover_the_four_quantities(make: Factories) -> None:
    spec = make.spec()
    solution = make.solution(spec, buy=np.array([0.1, 0.0, 0.0]), sell=np.array([0.0, 0.2, 0.0]))
    assert vector_values(solution, "w") is solution.w
    assert vector_values(solution, "trade").tolist() == [0.1, 0.2, 0.0]
    assert starting_values(spec, "w") is spec.w0
    assert starting_values(spec, "buy").tolist() == [0.0, 0.0, 0.0]


def test_check_against_spec_collects_every_missing_column_flag_scalar_and_group(make: Factories) -> None:
    with_adv = make.spec()
    spec = replace(with_adv, columns={name: values for name, values in with_adv.columns.items() if name != "adv_capacity"})
    constraints = (
        WeightLimit(name="cap", direction="<=", bounds=Decimal("0.4"), scope="no_such_flag"),
        ExposureLimit(name="beta_cap", direction="<=", column="beta", bounds=Decimal(1)),
        GroupLimit(name="bands", direction="<=", column="sector", bounds={"ENERGY": Decimal("0.5")}),
        CashLimit(name="cash", direction="<=", bounds=ScalarRef(scalar="no_such_scalar")),
        ParticipationLimit(name="adv", direction="<="),
    )
    problems = list(check_against_spec(constraints, spec))
    assert len(problems) == 5
    assert problems[0].startswith("cap:") and "no_such_flag" in problems[0]
    assert problems[1].startswith("beta_cap:") and "beta" in problems[1]
    assert problems[2].startswith("bands:") and "ENERGY" in problems[2]
    assert problems[3].startswith("cash:") and "no_such_scalar" in problems[3]
    assert problems[4].startswith("adv:") and "adv" in problems[4]
