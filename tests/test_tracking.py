from src.tracking import DEFAULT_URI, tracking_uri


def test_default_is_sqlite_in_project_root(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking_uri() == DEFAULT_URI
    assert DEFAULT_URI.startswith("sqlite:///") and DEFAULT_URI.endswith("mlflow.db")


def test_legacy_file_store_is_replaced(monkeypatch, capsys):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    assert tracking_uri() == DEFAULT_URI
    assert "Corrigez votre .env" in capsys.readouterr().out


def test_custom_database_or_server_is_kept(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    assert tracking_uri() == "http://127.0.0.1:5000"
