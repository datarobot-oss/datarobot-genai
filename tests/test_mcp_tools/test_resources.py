"""Tests for standard MCP resource handlers.

These tests verify the resource module structure and that handlers
return the correct shape when DR API is mocked.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestDatasetResourceShape:
    @pytest.mark.asyncio
    async def test_list_datasets_returns_list_of_dicts(self):
        mock_client = MagicMock()
        mock_client.get.return_value.json.return_value = {
            "data": [
                {"id": "ds-1", "name": "Sales Data"},
                {"id": "ds-2", "name": "Customer Data"},
            ]
        }
        with patch("datarobot.Client", return_value=mock_client):
            from datarobot_genai.drmcp.core.resources.datasets import list_datasets

            result = await list_datasets.fn()

        assert len(result) == 2
        assert result[0]["id"] == "ds-1"
        assert result[0]["uri"] == "dataset://ds-1"
        assert "name" in result[0]

    @pytest.mark.asyncio
    async def test_get_dataset_returns_metadata(self):
        import pandas as pd

        mock_dataset = MagicMock()
        mock_dataset.id = "ds-1"
        mock_dataset.name = "Sales Data"
        mock_dataset.created_at = "2024-01-01"
        mock_dataset.row_count = 1000
        mock_dataset.get_as_dataframe.return_value = pd.DataFrame(
            {"col_a": [1, 2], "col_b": ["x", "y"]}
        )

        with patch("datarobot.Dataset.get", return_value=mock_dataset):
            from datarobot_genai.drmcp.core.resources.datasets import get_dataset

            result = await get_dataset.fn("ds-1")

        assert result["id"] == "ds-1"
        assert result["name"] == "Sales Data"
        assert "columns" in result
        assert "sample_rows" in result
        assert len(result["sample_rows"]) == 2

    @pytest.mark.asyncio
    async def test_get_dataset_handles_dataframe_error(self):
        mock_dataset = MagicMock()
        mock_dataset.id = "ds-err"
        mock_dataset.name = "Bad Dataset"
        mock_dataset.created_at = "2024-01-01"
        mock_dataset.row_count = None
        mock_dataset.get_as_dataframe.side_effect = RuntimeError("network error")

        with patch("datarobot.Dataset.get", return_value=mock_dataset):
            from datarobot_genai.drmcp.core.resources.datasets import get_dataset

            result = await get_dataset.fn("ds-err")

        assert result["columns"] == []
        assert result["sample_rows"] == []
        assert "sample_error" in result


class TestDeploymentResourceShape:
    @pytest.mark.asyncio
    async def test_list_deployments_returns_list(self):
        mock_dep = MagicMock()
        mock_dep.id = "dep-1"
        mock_dep.label = "My Deployment"
        mock_dep.status = "active"

        with patch("datarobot.Deployment.list", return_value=[mock_dep]):
            from datarobot_genai.drmcp.core.resources.deployments import list_deployments

            result = await list_deployments.fn()

        assert len(result) == 1
        assert result[0]["id"] == "dep-1"
        assert result[0]["uri"] == "deployment://dep-1"

    @pytest.mark.asyncio
    async def test_get_deployment_returns_info(self):
        mock_dep = MagicMock()
        mock_dep.id = "dep-1"
        mock_dep.label = "My Deployment"
        mock_dep.model = {"id": "m-1", "project_id": "p-1", "type": "Gradient Boosted Trees"}

        mock_project = MagicMock()
        mock_project.target = "revenue"

        mock_dts = MagicMock()
        mock_dts.get.side_effect = Exception("not a time series project")

        with (
            patch("datarobot.Deployment.get", return_value=mock_dep),
            patch("datarobot.Project.get", return_value=mock_project),
            patch("datarobot.DatetimePartitioningSpecification", mock_dts),
            patch("datarobot.Model.get", side_effect=Exception("skip features")),
        ):
            from datarobot_genai.drmcp.core.resources.deployments import get_deployment

            result = await get_deployment.fn("dep-1")

        assert result["id"] == "dep-1"
        assert result["target"] == "revenue"
        assert result["is_time_series"] is False


class TestModelResourceShape:
    @pytest.mark.asyncio
    async def test_list_registered_models_returns_list(self):
        mock_client = MagicMock()
        mock_client.get.return_value.json.return_value = {
            "data": [{"id": "rm-1", "name": "Revenue Model", "target": "revenue"}]
        }

        with patch("datarobot.Client", return_value=mock_client):
            from datarobot_genai.drmcp.core.resources.models import list_registered_models

            result = await list_registered_models.fn()

        assert len(result) == 1
        assert result[0]["uri"] == "model://rm-1"

    @pytest.mark.asyncio
    async def test_get_registered_model_returns_details(self):
        mock_client = MagicMock()

        def mock_get(path, **kwargs):
            resp = MagicMock()
            if path == "registeredModels/rm-1/":
                resp.json.return_value = {
                    "id": "rm-1",
                    "name": "Revenue Model",
                    "target": "revenue",
                    "createdAt": "2024-01-01",
                }
            else:
                resp.json.return_value = {
                    "data": [{"id": "v-1", "modelVersionNumber": 1, "createdAt": "2024-01-01"}]
                }
            return resp

        mock_client.get.side_effect = mock_get

        with patch("datarobot.Client", return_value=mock_client):
            from datarobot_genai.drmcp.core.resources.models import get_registered_model

            result = await get_registered_model.fn("rm-1")

        assert result["id"] == "rm-1"
        assert result["target"] == "revenue"
        assert len(result["versions"]) == 1
