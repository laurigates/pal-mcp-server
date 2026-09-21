#!/usr/bin/env python3
"""
Cross-Tool Continuation Test

Tests comprehensive cross-tool continuation scenarios to ensure
conversation context is maintained when switching between different tools.
"""

from .conversation_base_test import ConversationBaseTest


class CrossToolContinuationTest(ConversationBaseTest):
    """Test comprehensive cross-tool continuation scenarios"""

    provider_agnostic = True

    @property
    def test_name(self) -> str:
        return "cross_tool_continuation"

    @property
    def test_description(self) -> str:
        return "Cross-tool conversation continuation scenarios"

    def run_test(self) -> bool:
        """Test comprehensive cross-tool continuation scenarios"""
        try:
            self.logger.info("🔧 Test: Cross-tool continuation scenarios")

            # Setup test environment for conversation testing
            self.setUp()

            scenarios = [
                ("chat -> thinkdeep -> codereview", self._test_chat_thinkdeep_codereview),
                ("analyze -> debug -> thinkdeep", self._test_analyze_debug_thinkdeep),
                ("multi-file cross-tool", self._test_multi_file_continuation),
            ]

            # Every scenario threads a continuation_id from one tool call to the
            # next, so they all have to reach the same server process.
            results = {}
            for name, scenario in scenarios:
                with self.server_session():
                    results[name] = scenario()

            for name, passed in results.items():
                self.logger.info(f"  {'✅' if passed else '❌'} {name}")

            passed_count = sum(results.values())
            self.logger.info(f"  Cross-tool continuation: {passed_count}/{len(scenarios)} scenarios passed")

            # Every scenario must pass. Accepting `passed_count > 0` is what let
            # two of these three fail behind a green line (issue #132).
            return passed_count == len(scenarios)

        except Exception as e:
            self.logger.error(f"Cross-tool continuation test failed: {e}")
            return False
        finally:
            self.cleanup_test_files()

    def _test_chat_thinkdeep_codereview(self) -> bool:
        """Test chat -> thinkdeep -> codereview scenario"""
        try:
            self.logger.info("  1: Testing chat -> thinkdeep -> codereview")

            # Start with chat
            chat_response, chat_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Please use low thinking mode. Look at this Python code and tell me what you think about it",
                    "absolute_file_paths": [self.test_files["python"]],
                    "model": self.simulator_model,
                },
            )

            if not chat_response or not chat_id:
                self.logger.error("Failed to start chat conversation")
                return False

            # Continue with thinkdeep
            thinkdeep_response, _ = self.call_mcp_tool(
                "thinkdeep",
                {
                    "step": "Think deeply about potential performance issues in this code. Please use low thinking mode.",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Building on previous chat analysis to examine performance issues",
                    "relevant_files": [self.test_files["python"]],  # Same file should be deduplicated
                    "continuation_id": chat_id,
                    "model": self.simulator_model,
                },
            )

            if not thinkdeep_response:
                self.logger.error("Failed chat -> thinkdeep continuation")
                return False

            # Continue with codereview
            codereview_response, _ = self.call_mcp_tool(
                "codereview",
                {
                    "step": "Building on our previous analysis, provide a comprehensive code review",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Continuing from previous chat and thinkdeep analysis for comprehensive review",
                    "relevant_files": [self.test_files["python"]],  # Same file should be deduplicated
                    "continuation_id": chat_id,
                    "model": self.simulator_model,
                },
            )

            if not codereview_response:
                self.logger.error("Failed thinkdeep -> codereview continuation")
                return False

            self.logger.info("  ✅ chat -> thinkdeep -> codereview working")
            return True

        except Exception as e:
            self.logger.error(f"Chat -> thinkdeep -> codereview scenario failed: {e}")
            return False

    def _test_analyze_debug_thinkdeep(self) -> bool:
        """Test analyze -> debug -> thinkdeep scenario"""
        try:
            self.logger.info("  2: Testing analyze -> debug -> thinkdeep")

            # Start with analyze
            analyze_response, analyze_id = self.call_mcp_tool(
                "analyze",
                {
                    "step": "Analyze this code for quality and performance issues",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Starting analysis of Python code for quality and performance issues",
                    "relevant_files": [self.test_files["python"]],
                    "model": self.simulator_model,
                },
            )

            if not analyze_response or not analyze_id:
                self.logger.warning("Failed to start analyze conversation, skipping scenario 2")
                return False

            # Continue with debug
            debug_response, _ = self.call_mcp_tool(
                "debug",
                {
                    "step": "Based on our analysis, help debug the performance issue in fibonacci",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Building on previous analysis to debug specific performance issue",
                    "relevant_files": [self.test_files["python"]],  # Same file should be deduplicated
                    "continuation_id": analyze_id,
                    "model": self.simulator_model,
                },
            )

            if not debug_response:
                self.logger.warning("  ⚠️ analyze -> debug continuation failed")
                return False

            # Continue with thinkdeep
            final_response, _ = self.call_mcp_tool(
                "thinkdeep",
                {
                    "step": "Think deeply about the architectural implications of the issues we've found. Please use low thinking mode.",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Building on analysis and debug findings to explore architectural implications",
                    "relevant_files": [self.test_files["python"]],  # Same file should be deduplicated
                    "continuation_id": analyze_id,
                    "model": self.simulator_model,
                },
            )

            if not final_response:
                self.logger.warning("  ⚠️ debug -> thinkdeep continuation failed")
                return False

            self.logger.info("  ✅ analyze -> debug -> thinkdeep working")
            return True

        except Exception as e:
            self.logger.error(f"Analyze -> debug -> thinkdeep scenario failed: {e}")
            return False

    def _test_multi_file_continuation(self) -> bool:
        """Test multi-file cross-tool continuation"""
        try:
            self.logger.info("  3: Testing multi-file cross-tool continuation")

            # Start with both files
            multi_response, multi_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Please use low thinking mode. Analyze both the Python code and configuration file",
                    "absolute_file_paths": [self.test_files["python"], self.test_files["config"]],
                    "model": self.simulator_model,
                },
            )

            if not multi_response or not multi_id:
                self.logger.warning("Failed to start multi-file conversation, skipping scenario 3")
                return False

            # Switch to codereview with same files (should use conversation history)
            multi_review, _ = self.call_mcp_tool(
                "codereview",
                {
                    "step": "Review both files in the context of our previous discussion",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Continuing multi-file analysis with code review perspective",
                    "relevant_files": [self.test_files["python"], self.test_files["config"]],  # Same files
                    "continuation_id": multi_id,
                    "model": self.simulator_model,
                },
            )

            if not multi_review:
                self.logger.warning("  ⚠️ Multi-file cross-tool continuation failed")
                return False

            self.logger.info("  ✅ Multi-file cross-tool continuation working")
            return True

        except Exception as e:
            self.logger.error(f"Multi-file continuation scenario failed: {e}")
            return False
