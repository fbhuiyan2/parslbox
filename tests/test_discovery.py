"""Systems and apps are discoverable, and the MCP descriptions say the truth.

Before this, the only way to learn a valid system name from the API or MCP was
a hand-typed list inside five Field descriptions. They had already drifted:
`orca` was missing from the app list, so an MCP caller could not learn the app
existed.
"""
import re

import pytest

from parslbox.apps.app_registry import APP_FACTORY
from parslbox.mcp import schemas
from parslbox.system_configs.loader import get_available_systems


def _quoted(text):
    return set(re.findall(r"'([a-z0-9-]+)'", text))


class TestSchemaDescriptionsAreGenerated:
    """Each of these was a hand-typed list. The point of the change is that
    adding a system or an app cannot leave one of them behind."""

    @pytest.mark.parametrize("schema,field", [
        (schemas.AddJobSchema, "config"),
        (schemas.QSubSchema, "config"),
        (schemas.SBatchSchema, "config"),
        (schemas.ResourceRequirementsSchema, "system"),
        (schemas.CommandLinesSchema, "system"),
    ])
    def test_the_field_names_every_system(self, schema, field):
        desc = schema.model_fields[field].description
        assert _quoted(desc) == set(get_available_systems()), field

    def test_the_app_field_names_every_builtin(self):
        desc = schemas.AddJobSchema.model_fields["app"].description
        assert set(APP_FACTORY) <= _quoted(desc)

    def test_orca_is_named(self):
        # The regression this change fixes: orca shipped in APP_FACTORY but
        # was absent from the description, so no agent could learn of it.
        assert "orca" in schemas.AddJobSchema.model_fields["app"].description

    def test_the_app_field_points_at_the_discovery_tool_for_custom_apps(self):
        desc = schemas.AddJobSchema.model_fields["app"].description
        assert "list_systems_and_apps" in desc


class TestTheMcpTool:
    def test_it_reports_every_system_one_per_line(self):
        from parslbox.mcp.mcp_server import list_systems_and_apps
        out = list_systems_and_apps()
        for name in get_available_systems():
            assert re.search(rf"^{re.escape(name)}: ", out, re.M), name

    def test_it_names_the_scheduler_so_a_user_can_choose(self):
        from parslbox.mcp.mcp_server import list_systems_and_apps
        out = list_systems_and_apps()
        assert "PBS" in out and "SLURM" in out

    def test_it_reports_both_app_kinds(self):
        from parslbox.mcp.mcp_server import list_systems_and_apps
        out = list_systems_and_apps()
        assert "apps builtin: " in out
        assert "apps custom: " in out

    def test_it_is_announced_in_the_server_instructions(self):
        # The capabilities list in the instructions block is hand-maintained
        # alongside the registrations; a tool missing from it is invisible.
        from parslbox.mcp import mcp_server
        assert "list_systems_and_apps" in mcp_server.mcp.instructions
