# SPDX-License-Identifier: Apache-2.0

"""The SSH probe must not open undeclared listeners."""

from sparksi.monitor.peer import remote_script


def test_metrics_follow_a_models_body_and_skip_undeclared_ports():
    script = remote_script()
    models = script.index("/v1/models")
    metrics = script.index("/metrics")
    assert models < metrics
    guard = script.index('if [ -n "$mj" ]')
    assert guard < metrics
    assert 'case " $http_ports "' in script
    assert "198\\.18\\.0\\." in script
    # The old probe curled every listener. Metrics now sit inside the models guard.
    assert script.count("/metrics") == 1
    after = script[guard:]
    assert "/metrics" in after
    assert "gloo" in script or "TCPStore" in script
