from fly.body.fly_body import FlyBody


def make_channels(
    *,
    turn=0.0,
    turn_bias=0.0,
    locomotion_drive=None,
    escape_takeoff=0.0,
    escape_long_mode=0.0,
    stop_freeze=0.0,
    backward_walk=0.0,
):
    channels = {
        "turn": turn,
        "turn_bias": turn_bias,
        "escape_takeoff": escape_takeoff,
        "escape_long_mode": escape_long_mode,
        "stop_freeze": stop_freeze,
        "backward_walk": backward_walk,
    }

    if locomotion_drive is not None:
        channels["locomotion_drive"] = locomotion_drive

    return channels


# ============================================================================
# BASIC STATE
# ============================================================================

def test_body_starts_at_rest():
    body = FlyBody()

    assert body.state.x_mm == 0.0
    assert body.state.y_mm == 0.0
    assert body.state.z_mm == 0.0
    assert body.state.speed_mm_s == 0.0
    assert body.state.turn_rate_deg_s == 0.0
    assert body.state.behaviour == "resting"


# ============================================================================
# LOCOMOTION
# ============================================================================

def test_body_walks_with_locomotion_drive():
    body = FlyBody()

    body.update(
        dt_ms=100.0,
        channels=make_channels(
            locomotion_drive=0.8,
            turn_bias=0.0,
        ),
        t_ms=100.0,
    )

    assert body.state.speed_mm_s > 0.0
    assert body.state.x_mm > 0.0
    assert body.state.behaviour == "walking"


def test_body_walks_from_turn_drive():
    body = FlyBody()

    body.update(
        dt_ms=100.0,
        channels=make_channels(
            turn=0.5,
            turn_bias=0.0,
        ),
        t_ms=100.0,
    )

    assert body.state.speed_mm_s > 0.0
    assert body.state.x_mm > 0.0


def test_body_turns():
    body = FlyBody()

    body.update(
        dt_ms=100.0,
        channels=make_channels(
            locomotion_drive=0.8,
            turn_bias=0.5,
        ),
        t_ms=100.0,
    )

    assert body.state.speed_mm_s > 0.0
    assert body.state.turn_rate_deg_s != 0.0
    assert body.state.heading_deg != 0.0


# ============================================================================
# OTHER MOTOR BEHAVIOURS
# ============================================================================

def test_body_freezes():
    body = FlyBody()

    body.update(
        dt_ms=100.0,
        channels=make_channels(
            locomotion_drive=0.8,
            stop_freeze=1.0,
        ),
        t_ms=100.0,
    )

    assert body.state.speed_mm_s == 0.0
    assert body.state.behaviour == "freezing (DNp09)"


def test_body_backward_walk():
    body = FlyBody()

    body.update(
        dt_ms=100.0,
        channels=make_channels(
            locomotion_drive=0.8,
            backward_walk=1.0,
        ),
        t_ms=100.0,
    )

    assert body.state.speed_mm_s < 0.0
    assert body.state.behaviour == "walking backward (MDN)"


# ============================================================================
# SHORT GF ESCAPE
# ============================================================================

def test_body_escape_takeoff():
    body = FlyBody()

    # Command first appears at t=10 ms.
    body.update(
        dt_ms=10.0,
        channels=make_channels(
            escape_takeoff=1.0,
        ),
        t_ms=10.0,
    )

    # It should be armed but not yet airborne.
    assert body.state.airborne is False
    assert body.state.escape_mode == "short"

    # 10 ms later -> 10 ms elapsed since command.
    # This exceeds the 5 ms GF latency.
    body.update(
        dt_ms=10.0,
        channels=make_channels(
            escape_takeoff=1.0,
        ),
        t_ms=20.0,
    )

    assert body.state.airborne is True
    assert body.state.escape_mode == "short"
    assert body.state.speed_mm_s == 750.0


# ============================================================================
# LONG ESCAPE
# ============================================================================

def test_body_long_mode_escape():
    body = FlyBody()

    # Command first appears at t=50 ms.
    body.update(
        dt_ms=50.0,
        channels=make_channels(
            escape_long_mode=1.0,
        ),
        t_ms=50.0,
    )

    assert body.state.airborne is False
    assert body.state.escape_mode == "long"

    # 100 ms elapsed.
    body.update(
        dt_ms=100.0,
        channels=make_channels(
            escape_long_mode=1.0,
        ),
        t_ms=150.0,
    )

    assert body.state.airborne is False
    assert body.state.escape_mode == "long"
    assert body.state.wing_angle_deg > 0.0
    assert body.state.leg_extension > 0.0

    # Advance to t=250 ms.
    # Since the command started at t=50 ms, exactly 200 ms
    # of preparation have elapsed.
    #
    # The body internally integrates this 100 ms in 5 ms
    # physics steps, preventing artificial immediate landing.
    body.update(
        dt_ms=100.0,
        channels=make_channels(
            escape_long_mode=1.0,
        ),
        t_ms=250.0,
    )

    assert body.state.airborne is True
    assert body.state.escape_mode == "long"


# ============================================================================
# RESET
# ============================================================================

def test_body_reset():
    body = FlyBody()

    body.update(
        dt_ms=100.0,
        channels=make_channels(
            locomotion_drive=1.0,
            turn_bias=0.5,
        ),
        t_ms=100.0,
    )

    assert body.state.speed_mm_s > 0.0
    assert body.state.heading_deg != 0.0

    body.reset()

    assert body.state.x_mm == 0.0
    assert body.state.y_mm == 0.0
    assert body.state.z_mm == 0.0
    assert body.state.speed_mm_s == 0.0
    assert body.state.turn_rate_deg_s == 0.0
    assert body.state.heading_deg == 0.0
    assert body.state.behaviour == "resting"
    assert body.state.airborne is False


# ============================================================================
# ENVIRONMENT INDEPENDENCE
# ============================================================================

def test_body_never_requires_environment():
    body = FlyBody()

    body.update(
        dt_ms=100.0,
        channels=make_channels(
            turn=0.5,
            turn_bias=0.0,
        ),
        t_ms=100.0,
    )

    assert body.state.speed_mm_s > 0.0
    assert body.state.x_mm > 0.0


# ============================================================================
# PROVENANCE
# ============================================================================

def test_body_provenance():
    body = FlyBody()

    provenance = body.provenance

    assert "body model" in provenance["role"]
    assert "descending-neuron" in provenance["inputs"]
    assert "stimulus" in provenance["inputs"]
    assert "locomotion_caveat" in provenance