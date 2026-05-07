from pathlib import Path
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--discovery",
        action="store_true",
        default=False,
        help="Run discovery mode instead of equation comparison",
    )
    parser.addoption(
        "--report",
        action="store_true",
        default=False,
        help="Save discovery report files",
    )
    parser.addoption(
        "--report-dir",
        action="store",
        default="reports",
        help="Base directory for reports",
    )
    parser.addoption(
        "--operators",
        action="store",
        default="DeepXDEBasedFitness,PIC,L2LRFitness",
        help="Comma-separated list of operators to test",
    )

@pytest.fixture
def runtime_options(request):
    operators = request.config.getoption("--operators")
    operator_list = [op.strip() for op in operators.split(",") if op.strip()]
    return {
        "discovery": request.config.getoption("--discovery"),
        "report": request.config.getoption("--report"),
        "report_dir": Path(request.config.getoption("--report-dir")),
        "operators": operator_list,
    }

def pytest_configure(config):
    config.addinivalue_line("markers", "functional: functional tests")
    config.addinivalue_line("markers", "discovery: discovery tests")