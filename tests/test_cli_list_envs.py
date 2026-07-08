"""Tests for the `cotter list-envs` subcommand."""

import pytest

from cotter.cli import main


class TestListEnvs:
    def test_lists_core_and_extension_envs(self, capsys):
        rc = main(["list-envs"])
        out = capsys.readouterr().out
        assert rc == 0
        # core MuJoCo (gymnasium) and an extension env (gymnasium_robotics)
        assert "InvertedPendulum-v5" in out
        assert "gymnasium" in out
        assert "env id(s)" in out

    def test_filter_narrows_output(self, capsys):
        rc = main(["list-envs", "--filter", "Fetch"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "FetchPickAndPlace-v4" in out
        assert "matching 'Fetch'" in out
        # a non-Fetch env must be filtered out
        assert "InvertedPendulum-v5" not in out

    def test_filter_is_case_insensitive(self, capsys):
        rc = main(["list-envs", "--filter", "ant"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Ant-v5" in out

    def test_no_matches_is_graceful(self, capsys):
        rc = main(["list-envs", "--filter", "definitely-not-an-env-xyz"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "no envs" in out

    def test_grouped_by_package(self, capsys):
        main(["list-envs", "--filter", "Fetch"])
        out = capsys.readouterr().out
        # Fetch envs come from gymnasium_robotics
        assert "gymnasium_robotics" in out
