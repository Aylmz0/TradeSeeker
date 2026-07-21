"""Tests for src/services/alert_system.py."""

import pytest

from src.services.alert_system import Alert, AlertLevel, AlertManager, AlertType


class TestAlertCreation:
    """Tests for Alert dataclass creation."""

    def test_alert_creation(self):
        """Alert dataclass can be created with required fields."""
        alert = Alert(
            id="alert_1",
            type=AlertType.PRICE_MOVEMENT,
            level=AlertLevel.WARNING,
            title="Price Alert",
            message="ETH moved up 5%",
        )
        assert alert.id == "alert_1"
        assert alert.type == AlertType.PRICE_MOVEMENT
        assert alert.level == AlertLevel.WARNING
        assert alert.title == "Price Alert"
        assert alert.message == "ETH moved up 5%"
        assert alert.symbol is None
        assert alert.data is None

    def test_alert_creation_with_optional_fields(self):
        """Alert can be created with optional fields."""
        alert = Alert(
            id="alert_2",
            type=AlertType.RISK_LIMIT,
            level=AlertLevel.CRITICAL,
            title="Risk Breach",
            message="Max drawdown exceeded",
            symbol="BTC",
            data={"drawdown": 0.12},
        )
        assert alert.symbol == "BTC"
        assert alert.data == {"drawdown": 0.12}
        assert alert.timestamp is not None

    def test_alert_to_dict(self):
        """Alert to_dict produces correct dictionary."""
        alert = Alert(
            id="alert_3",
            type=AlertType.PERFORMANCE,
            level=AlertLevel.INFO,
            title="Profit",
            message="Up 10%",
        )
        d = alert.to_dict()
        assert d["id"] == "alert_3"
        assert d["type"] == "performance"
        assert d["level"] == "INFO"
        assert "timestamp" in d


class TestAlertSeverity:
    """Tests for AlertLevel severity values."""

    def test_alert_severity_info(self):
        """INFO is a valid severity level."""
        assert AlertLevel.INFO.value == "INFO"

    def test_alert_severity_warning(self):
        """WARNING is a valid severity level."""
        assert AlertLevel.WARNING.value == "WARNING"

    def test_alert_severity_critical(self):
        """CRITICAL is a valid severity level."""
        assert AlertLevel.CRITICAL.value == "CRITICAL"

    def test_alert_severity_all_levels(self):
        """All severity levels are covered."""
        assert set(AlertLevel) == {AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL}


class TestAlertManager:
    """Tests for AlertManager instantiation."""

    def test_alert_manager(self):
        """AlertManager can be instantiated."""
        manager = AlertManager()
        assert manager.alerts == []
        assert manager.max_alerts == 100
        assert manager.alert_handlers == []

    def test_alert_manager_create_alert(self):
        """AlertManager can create alerts."""
        manager = AlertManager()
        alert = manager.create_alert(
            alert_type=AlertType.SYSTEM,
            level=AlertLevel.INFO,
            title="Test",
            message="Test message",
        )
        assert alert in manager.alerts
        assert alert.level == AlertLevel.INFO

    def test_alert_manager_max_alerts(self):
        """AlertManager caps alerts at max_alerts."""
        manager = AlertManager()
        manager.max_alerts = 3
        for i in range(5):
            manager.create_alert(
                alert_type=AlertType.SYSTEM,
                level=AlertLevel.INFO,
                title=f"Alert {i}",
                message=f"Message {i}",
            )
        assert len(manager.alerts) == 3
        assert manager.alerts[0].title == "Alert 2"
