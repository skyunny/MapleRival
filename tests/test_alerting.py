import unittest

from maplerival.alerting import (
    OWN_CLOSE,
    OWN_SAFE,
    RIVAL_CLOSE,
    RIVAL_SAFE,
    classify_gap,
    progress_score,
    transition_message,
)
from maplerival.multi_alerting import (
    decrypt_webhook,
    encrypt_webhook,
    rival_added_message,
    rival_removed_message,
    webhook_registered_message,
)


class AlertStateTest(unittest.TestCase):
    def test_four_gap_states(self):
        self.assertEqual(classify_gap(1.0), OWN_SAFE)
        self.assertEqual(classify_gap(0.999), OWN_CLOSE)
        self.assertEqual(classify_gap(-0.999), RIVAL_CLOSE)
        self.assertEqual(classify_gap(-1.0), RIVAL_SAFE)

    def test_level_is_part_of_progress_score(self):
        self.assertEqual(progress_score(296, 5.0) - progress_score(295, 95.0), 10.0)

    def test_overtake_message_names_the_rival(self):
        message = transition_message(OWN_CLOSE, RIVAL_CLOSE, "탕무스", "꽃게쥬", -0.2)
        self.assertIn("꽃게쥬님에게 역전당했어요", message)

    def test_webhook_is_encrypted_at_rest(self):
        url = "https://discord.com/api/webhooks/example/token"
        encrypted = encrypt_webhook("server-secret", url)
        self.assertNotIn("discord.com", encrypted)
        self.assertEqual(decrypt_webhook("server-secret", encrypted), url)

    def test_monitoring_event_messages(self):
        self.assertEqual(
            webhook_registered_message(["A", "B", "C"]),
            "MAPLE RIVAL : 라이벌 경험치 모니터링을 시작합니다.\n현재 모니터링 중인 라이벌 캐릭터 : A, B, C",
        )
        self.assertIn("모니터링 중인 라이벌 캐릭터가 없습니다", webhook_registered_message([]))
        self.assertEqual(
            rival_added_message("C", ["A", "B", "C"]),
            "C 경험치 모니터링을 시작합니다.\n현재 모니터링 중인 라이벌 캐릭터 : A, B, C",
        )
        self.assertEqual(
            rival_removed_message("C", ["A", "B"]),
            "C 경험치 모니터링을 종료합니다.\n현재 모니터링 중인 라이벌 캐릭터 : A, B",
        )


if __name__ == "__main__":
    unittest.main()
