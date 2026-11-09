import numpy as np
import pytest

from swarm.config import ObservationConfig
from swarm.obs.local_window import (
    CHANNEL_OCCUPANCY,
    CHANNEL_OUT_OF_BOUNDS,
    CHANNEL_TARGET,
    ObservationBuilder,
    build_spec,
)
from swarm.sim import moves

from .helpers import random_state, state_from_cells

GRID = (10, 10)


def builder_for(grid_shape=GRID, **overrides) -> ObservationBuilder:
    config = ObservationConfig(**overrides)
    return ObservationBuilder(config, grid_shape)


# -- the spec ----------------------------------------------------------


def test_spec_lists_its_fields_in_order():
    spec = build_spec(ObservationConfig(k_nearest_targets=2))
    assert spec.field_names == ("target_0", "target_1", "on_target", "relative_position")
    assert spec.vector_size == 4 + 4 + 1 + 2


def test_spec_tracks_optional_fields():
    spec = build_spec(
        ObservationConfig(
            k_nearest_targets=1,
            use_relative_position=False,
            use_absolute_position=True,
            use_module_tag=True,
            use_global_map=True,
            global_map_size=4,
        )
    )
    assert spec.field_names == (
        "target_0",
        "on_target",
        "absolute_position",
        "module_tag",
        "global_map",
    )
    assert spec.vector_size == 4 + 1 + 2 + 1 + 16


def test_spec_slices_address_the_right_columns():
    spec = build_spec(ObservationConfig(k_nearest_targets=3))
    assert spec.field_slice("target_0") == slice(0, 4)
    assert spec.field_slice("target_2") == slice(8, 12)
    assert spec.field_slice("on_target") == slice(12, 13)
    assert spec.field_slice("relative_position") == slice(13, 15)


def test_spec_rejects_an_unknown_field():
    with pytest.raises(KeyError, match="unknown observation field"):
        build_spec(ObservationConfig()).field_slice("nope")


# -- shapes and types --------------------------------------------------


def test_observation_shapes_match_the_spec():
    builder = builder_for()
    state = state_from_cells([(2, 2), (3, 2), (3, 3)], GRID, target=[(5, 5), (5, 6), (6, 6)])
    obs = builder.build(state)

    assert obs["window"].shape == (3, *builder.spec.window_shape)
    assert obs["vector"].shape == (3, builder.spec.vector_size)
    assert obs["mask"].shape == (3, moves.N_ACTIONS)
    assert obs["window"].dtype == np.float32
    assert obs["vector"].dtype == np.float32
    assert obs["mask"].dtype == bool


def test_observation_values_are_finite_and_bounded():
    rng = np.random.default_rng(0)
    builder = builder_for()
    for _ in range(50):
        state = random_state(rng, n_modules=int(rng.integers(1, 12)), grid_shape=GRID)
        state.target[:] = False
        target_cells = random_state(rng, n_modules=4, grid_shape=GRID).positions
        state.target[target_cells[:, 0], target_cells[:, 1]] = True

        obs = builder.build(state)
        assert np.isfinite(obs["vector"]).all()
        assert np.isfinite(obs["window"]).all()
        assert np.abs(obs["vector"]).max() <= 1.0 + 1e-6


def test_mask_is_the_rules_mask():
    builder = builder_for()
    state = state_from_cells([(2, 2), (3, 2), (3, 3)], GRID, target=[(5, 5)])
    assert np.array_equal(builder.build(state)["mask"], moves.legal_action_mask(state))


def test_a_passed_in_mask_is_used_verbatim():
    """The caller can hand in a mask it already computed, to avoid double work."""
    builder = builder_for()
    state = state_from_cells([(2, 2)], GRID, target=[(5, 5)])
    supplied = np.zeros((1, moves.N_ACTIONS), dtype=bool)
    supplied[0, moves.STAY] = True
    assert np.array_equal(builder.build(state, supplied)["mask"], supplied)


# -- the local window --------------------------------------------------


def test_window_is_centred_on_each_module():
    builder = builder_for(window=5)
    state = state_from_cells([(5, 5), (5, 6)], GRID, target=[(5, 4)])
    window = builder.build(state)["window"]

    # Module 0 sits at the centre of its own window.
    assert window[0, CHANNEL_OCCUPANCY, 2, 2] == 1.0
    # Its neighbour is one cell to the right of centre.
    assert window[0, CHANNEL_OCCUPANCY, 2, 3] == 1.0
    assert window[0, CHANNEL_OCCUPANCY, 2, 1] == 0.0
    # And the target cell one to the left shows in the target channel.
    assert window[0, CHANNEL_TARGET, 2, 1] == 1.0


def test_window_marks_out_of_bounds_separately_from_empty():
    """A wall and an empty cell are different things when choosing a move."""
    builder = builder_for(window=5)
    state = state_from_cells([(0, 0)], GRID, target=[(4, 4)])
    window = builder.build(state)["window"]

    assert window[0, CHANNEL_OUT_OF_BOUNDS, 0, :].all()  # two rows above the grid
    assert window[0, CHANNEL_OUT_OF_BOUNDS, 1, :].all()
    assert not window[0, CHANNEL_OUT_OF_BOUNDS, 2, 2]  # the module's own cell
    assert window[0, CHANNEL_OUT_OF_BOUNDS, :, 0].all()  # two columns left of the grid
    # Out-of-bounds cells are not reported as occupied.
    assert window[0, CHANNEL_OCCUPANCY, 0, 0] == 0.0


def test_window_size_follows_the_config():
    state = state_from_cells([(5, 5)], GRID, target=[(1, 1)])
    for window in (3, 5, 7):
        obs = builder_for(window=window).build(state)
        assert obs["window"].shape == (1, 3, window, window)


# -- nearest unfilled targets ------------------------------------------


def test_nearest_targets_are_sorted_by_distance():
    builder = builder_for(k_nearest_targets=3)
    state = state_from_cells([(5, 5)], GRID, target=[(5, 6), (5, 8), (1, 5)])
    aux = builder.build(state)["aux"]

    assert aux["nearest_valid"][0].all()
    distances = np.abs(aux["nearest_delta"][0]).sum(axis=1)
    assert list(distances) == [1, 3, 4]


def test_a_target_cell_under_a_module_is_not_unfilled():
    """Only unfilled cells are candidates, so a module on target looks past its own cell."""
    builder = builder_for(k_nearest_targets=1)
    state = state_from_cells([(5, 5)], GRID, target=[(5, 5), (5, 9)])
    obs = builder.build(state)

    assert obs["aux"]["n_unfilled"] == 1
    assert list(obs["aux"]["nearest_delta"][0, 0]) == [0, 4]
    assert obs["vector"][0, builder.spec.field_slice("on_target")][0] == 1.0


def test_missing_targets_are_padded_and_flagged():
    builder = builder_for(k_nearest_targets=3)
    state = state_from_cells([(5, 5)], GRID, target=[(5, 7)])
    obs = builder.build(state)

    valid = obs["aux"]["nearest_valid"][0]
    assert list(valid) == [True, False, False]
    assert (obs["aux"]["nearest_delta"][0, 1:] == 0).all()

    # The validity flag is the last element of each slot.
    for index, expected in enumerate([1.0, 0.0, 0.0]):
        slot = obs["vector"][0, builder.spec.field_slice(f"target_{index}")]
        assert slot[3] == expected


def test_completed_target_leaves_every_slot_invalid():
    builder = builder_for(k_nearest_targets=2)
    state = state_from_cells([(5, 5), (5, 6)], GRID, target=[(5, 5), (5, 6)])
    obs = builder.build(state)
    assert obs["aux"]["n_unfilled"] == 0
    assert not obs["aux"]["nearest_valid"].any()
    assert (obs["aux"]["nearest_delta"] == 0).all()


def test_direction_is_split_across_the_axes():
    """Each valid slot encodes direction as a pair summing to 1 in absolute value."""
    builder = builder_for(k_nearest_targets=1)
    state = state_from_cells([(5, 5)], GRID, target=[(2, 8)])  # 3 up, 3 right
    slot = builder.build(state)["vector"][0, builder.spec.field_slice("target_0")]

    assert slot[0] == pytest.approx(-0.5)
    assert slot[1] == pytest.approx(0.5)
    assert abs(slot[0]) + abs(slot[1]) == pytest.approx(1.0)
    assert slot[2] == pytest.approx(6 / (GRID[0] + GRID[1]))


def test_distance_grows_with_separation():
    builder = builder_for(k_nearest_targets=1)
    near = state_from_cells([(5, 5)], GRID, target=[(5, 6)])
    far = state_from_cells([(5, 5)], GRID, target=[(5, 9)])
    index = builder.spec.field_slice("target_0")
    assert builder.build(near)["vector"][0, index][2] < builder.build(far)["vector"][0, index][2]


# -- position and optional fields --------------------------------------


def test_relative_position_is_centred_on_the_swarm():
    builder = builder_for()
    state = state_from_cells([(4, 5), (6, 5)], GRID, target=[(1, 1)])
    vector = builder.build(state)["vector"]
    index = builder.spec.field_slice("relative_position")

    # Centroid is (5, 5), so the two modules sit symmetrically about it.
    assert vector[0, index][0] == pytest.approx(-0.1)
    assert vector[1, index][0] == pytest.approx(0.1)
    assert vector[:, index][:, 1] == pytest.approx([0.0, 0.0])


def test_relative_position_does_not_depend_on_where_the_swarm_sits():
    """This is the point of it: the same shape reads the same anywhere on the grid."""
    builder = builder_for()
    index = builder.spec.field_slice("relative_position")
    here = state_from_cells([(2, 2), (3, 2)], GRID, target=[(9, 9)])
    there = state_from_cells([(6, 7), (7, 7)], GRID, target=[(9, 9)])
    assert builder.build(here)["vector"][:, index] == pytest.approx(
        builder.build(there)["vector"][:, index]
    )


def test_absolute_position_is_normalized():
    builder = builder_for(use_relative_position=False, use_absolute_position=True)
    state = state_from_cells([(5, 2)], GRID, target=[(1, 1)])
    vector = builder.build(state)["vector"]
    assert vector[0, builder.spec.field_slice("absolute_position")] == pytest.approx([0.5, 0.2])


def test_module_tags_are_per_module_and_per_episode():
    """Identical observations under a shared policy need something to break the tie."""
    builder = builder_for(use_module_tag=True)
    state = state_from_cells([(5, 5), (5, 6), (5, 7)], GRID, target=[(1, 1)])

    builder.reset(state.n_modules, np.random.default_rng(0))
    index = builder.spec.field_slice("module_tag")
    first = builder.build(state)["vector"][:, index].copy()
    assert np.unique(first).size == 3
    # Stable within an episode.
    assert builder.build(state)["vector"][:, index] == pytest.approx(first)

    builder.reset(state.n_modules, np.random.default_rng(1))
    assert not np.allclose(builder.build(state)["vector"][:, index], first)


def test_tagged_observations_require_a_reset():
    builder = builder_for(use_module_tag=True)
    state = state_from_cells([(5, 5)], GRID, target=[(1, 1)])
    with pytest.raises(RuntimeError, match="reset\\(\\) must be called"):
        builder.build(state)


def test_reset_is_a_no_op_without_tags():
    builder = builder_for()
    builder.reset(3, np.random.default_rng(0))
    state = state_from_cells([(5, 5)], GRID, target=[(1, 1)])
    builder.build(state)  # must not raise


def test_global_map_is_shared_by_every_module():
    builder = builder_for(use_global_map=True, global_map_size=5)
    state = state_from_cells([(2, 2), (2, 3)], GRID, target=[(8, 8), (8, 9)])
    vector = builder.build(state)["vector"]
    index = builder.spec.field_slice("global_map")

    rows = vector[:, index]
    assert rows.shape == (2, 25)
    assert np.array_equal(rows[0], rows[1])
    # Occupancy minus target: positive where the swarm is, negative at the target.
    assert rows[0].sum() == pytest.approx(0.0, abs=1e-6)
    assert rows[0].max() > 0 and rows[0].min() < 0


# -- determinism -------------------------------------------------------


def test_observations_are_deterministic():
    builder = builder_for()
    state = state_from_cells([(2, 2), (3, 2), (3, 3)], GRID, target=[(7, 7), (7, 8), (8, 8)])
    first, second = builder.build(state), builder.build(state)
    assert np.array_equal(first["vector"], second["vector"])
    assert np.array_equal(first["window"], second["window"])


def test_builder_does_not_mutate_the_state():
    builder = builder_for(use_global_map=True)
    state = state_from_cells([(2, 2), (3, 2)], GRID, target=[(7, 7), (7, 8)])
    occupancy, target, positions = (
        state.occupancy.copy(),
        state.target.copy(),
        state.positions.copy(),
    )
    builder.build(state)
    assert np.array_equal(state.occupancy, occupancy)
    assert np.array_equal(state.target, target)
    assert np.array_equal(state.positions, positions)


def test_single_module_swarm_is_handled():
    builder = builder_for()
    state = state_from_cells([(0, 0)], GRID, target=[(9, 9)])
    obs = builder.build(state)
    assert obs["vector"].shape == (1, builder.spec.vector_size)
    assert np.isfinite(obs["vector"]).all()
