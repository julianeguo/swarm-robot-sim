from swarm.render.ascii import (
    EMPTY,
    FILLED_TARGET,
    MODULE,
    UNFILLED_TARGET,
    render,
    render_with_summary,
    summary,
)

from .helpers import state_from_cells

GRID = (4, 4)


def test_renders_every_cell_category():
    state = state_from_cells([(1, 1), (1, 2)], GRID, target=[(1, 2), (2, 2)])
    lines = render(state).splitlines()

    assert lines[0] == "+----+"
    assert lines[-1] == "+----+"
    body = [line[1:-1] for line in lines[1:-1]]

    assert body[1][1] == MODULE  # module, not on a target cell
    assert body[1][2] == FILLED_TARGET  # module sitting on a target cell
    assert body[2][2] == UNFILLED_TARGET  # target cell with no module
    assert body[0][0] == EMPTY


def test_render_is_row_major_top_down():
    state = state_from_cells([(0, 0)], GRID)
    body = [line[1:-1] for line in render(state).splitlines()[1:-1]]
    assert body[0][0] == MODULE, "row 0 must render as the top line"


def test_crop_trims_to_the_interesting_bounding_box():
    state = state_from_cells([(1, 1), (1, 2)], (10, 10), target=[(1, 2)])
    cropped = render(state, crop=True).splitlines()
    assert len(cropped) == 3  # border, one row of content, border
    assert cropped[1] == f"|{MODULE}{FILLED_TARGET}|"


def test_crop_includes_target_cells_outside_the_swarm():
    state = state_from_cells([(1, 1)], (10, 10), target=[(5, 5)])
    cropped = render(state, crop=True).splitlines()
    assert len(cropped) == 2 + 5  # rows 1..5 inclusive


def test_summary_reports_progress():
    state = state_from_cells([(1, 1), (1, 2)], GRID, target=[(1, 2), (2, 2)])
    text = summary(state)
    assert "1/2 filled" in text
    assert "modules   2" in text


def test_summary_handles_no_target():
    assert "no target" in summary(state_from_cells([(1, 1)], GRID))


def test_render_with_summary_combines_both():
    state = state_from_cells([(1, 1)], GRID)
    combined = render_with_summary(state)
    assert combined.startswith(summary(state))
    assert render(state) in combined
