"""
Workflow Prompts for MCP Server

Guided workflows for common tasks.
"""

from typing import Any, Dict, List

from mcp.types import TextContent


class WorkflowPrompts:
    """
    Workflow prompt provider.

    Prompts:
    - /analyze-and-recommend: Full workflow from data upload to model recommendation
    - /explain-hard-problem: Deep dive into complex cases
    - /validate-recommendation: Run offline benchmark validation
    - /update-knowledge-base: Refine SystemPrompt scores from empirical feedback
    """

    def get_prompt_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP prompt definitions."""
        return [
            {
                "name": "analyze-and-recommend",
                "description": (
                    "Complete workflow: Analyze dataset → Detect stress factors → Recommend optimal model. "
                    "This prompt guides through the full process from data upload to getting actionable recommendations."
                ),
                "arguments": [
                    {
                        "name": "data_path",
                        "description": "Path to CSV or Parquet file to analyze",
                        "required": True
                    }
                ]
            },
            {
                "name": "explain-hard-problem",
                "description": (
                    "Deep dive into complex dataset characteristics. "
                    "Explains why a dataset is considered a 'hard problem' (severe skew, zipfian distribution, etc.) "
                    "and what models are best suited to handle these challenges."
                ),
                "arguments": [
                    {
                        "name": "dataset_id",
                        "description": "Dataset ID from previous analysis",
                        "required": True
                    }
                ]
            },
            {
                "name": "validate-recommendation",
                "description": (
                    "Run offline benchmark validation for a recommended model. "
                    "Generates synthetic control datasets and validates model performance "
                    "using Wasserstein Distance and TVD metrics."
                ),
                "arguments": [
                    {
                        "name": "dataset_id",
                        "description": "Dataset ID from previous analysis",
                        "required": True
                    },
                    {
                        "name": "model_name",
                        "description": "Model to validate",
                        "required": True
                    }
                ]
            },
            {
                "name": "update-knowledge-base",
                "description": (
                    "Update SystemPrompt capability scores based on empirical benchmark results. "
                    "Refines the recommendation engine's knowledge base when validation results "
                    "differ from theoretical expectations."
                ),
                "arguments": [
                    {
                        "name": "benchmark_results",
                        "description": "JSON object with validation metrics",
                        "required": True
                    }
                ]
            },
        ]

    async def get_prompt(self, name: str, arguments: Dict[str, str]) -> Dict[str, Any]:
        """Get a prompt with arguments filled in."""
        if name == "analyze-and-recommend":
            return await self._analyze_and_recommend(arguments)
        elif name == "explain-hard-problem":
            return await self._explain_hard_problem(arguments)
        elif name == "validate-recommendation":
            return await self._validate_recommendation(arguments)
        elif name == "update-knowledge-base":
            return await self._update_knowledge_base(arguments)
        else:
            raise ValueError(f"Unknown prompt: {name}")

    async def _analyze_and_recommend(self, arguments: Dict[str, str]) -> Dict[str, Any]:
        """Generate analyze-and-recommend workflow prompt."""
        data_path = arguments.get("data_path", "[data_path]")

        messages = [
            {
                "role": "user",
                "content": TextContent(
                    type="text",
                    text=f"""I need help selecting the best synthetic data generation model for my dataset.

Dataset: {data_path}

Please:
1. Analyze the dataset stress profile (skewness, cardinality, zipfian ratio, correlation)
2. Identify any "hard problems" (severe skew, high cardinality, zipfian distributions)
3. Recommend the optimal synthesis model based on dataset characteristics
4. Explain why this model is best suited for my data
5. Provide alternatives if the primary recommendation has limitations

Use the following tools in order:
- analyze_stress_profile: To understand dataset characteristics
- rank_models_hybrid: To get model recommendations
- explain_recommendation_reasoning: To generate detailed explanation
"""
                )
            }
        ]

        return {
            "messages": messages,
            "description": "Complete workflow from data analysis to model recommendation"
        }

    async def _explain_hard_problem(self, arguments: Dict[str, str]) -> Dict[str, Any]:
        """Generate explain-hard-problem workflow prompt."""
        dataset_id = arguments.get("dataset_id", "[dataset_id]")

        messages = [
            {
                "role": "user",
                "content": TextContent(
                    type="text",
                    text=f"""I want to understand why my dataset is considered a "hard problem" for synthesis.

Dataset ID: {dataset_id}

Please:
1. Load the cached dataset profile
2. Analyze stress factors:
   - Severe Skew (> 2.0): Long tail distributions
   - High Cardinality (> 500): Many unique values
   - Zipfian Distribution: Top 20% categories dominate
   - Small Data (< 500 rows): Overfitting risk
3. Explain which traditional models fail on these characteristics
4. Recommend specialized models that handle these challenges
5. Provide validation strategy to verify model performance

Use the following resources and tools:
- datasets://profiles/{dataset_id}: To load cached profile
- benchmarks://thresholds: To understand detection thresholds
- rank_models_hybrid: To see which models are suitable
- explain_recommendation_reasoning: For detailed explanation
"""
                )
            }
        ]

        return {
            "messages": messages,
            "description": "Deep dive into complex dataset characteristics"
        }

    async def _validate_recommendation(self, arguments: Dict[str, str]) -> Dict[str, Any]:
        """Generate validate-recommendation workflow prompt."""
        dataset_id = arguments.get("dataset_id", "[dataset_id]")
        model_name = arguments.get("model_name", "[model_name]")

        messages = [
            {
                "role": "user",
                "content": TextContent(
                    type="text",
                    text=f"""I want to validate the recommended model against benchmark datasets.

Dataset ID: {dataset_id}
Model: {model_name}

Please:
1. Load the dataset profile
2. Identify the primary stress factors (skew, cardinality, zipfian)
3. Generate appropriate benchmark datasets:
   - If severe skew: Generate 'long_tail' benchmark
   - If zipfian: Generate 'needle_haystack' benchmark
   - If small data: Generate 'small_data_trap' benchmark
4. Run validation (note: actual model training is external to Synthony)
5. Compare expected vs actual performance
6. Recommend knowledge base updates if needed

Use the following tools:
- datasets://profiles/{dataset_id}: To load profile
- generate_benchmark_dataset: To create test datasets
- benchmarks://results/{model_name}/{{dataset_type}}: To check existing results
"""
                )
            }
        ]

        return {
            "messages": messages,
            "description": "Offline benchmark validation workflow"
        }

    async def _update_knowledge_base(self, arguments: Dict[str, str]) -> Dict[str, Any]:
        """Generate update-knowledge-base workflow prompt."""
        benchmark_results = arguments.get("benchmark_results", "[benchmark_results]")

        messages = [
            {
                "role": "user",
                "content": TextContent(
                    type="text",
                    text=f"""I have new benchmark results that differ from expected model performance.

Benchmark Results: {benchmark_results}

Please:
1. Parse the benchmark results (Wasserstein Distance, TVD, R²)
2. Compare against current model capability scores in SystemPrompt
3. Identify discrepancies:
   - If model performed better than expected: Consider increasing score
   - If model performed worse: Consider decreasing score
4. Generate recommendations for SystemPrompt updates
5. Create a new system prompt version with updated scores

Use the following resources:
- guidelines://system-prompt: To see current capability scores
- benchmarks://thresholds: To understand validation criteria

Note: This workflow helps maintain the self-correcting feedback loop.
SystemPrompt scores should reflect empirical evidence, not just theoretical expectations.
"""
                )
            }
        ]

        return {
            "messages": messages,
            "description": "Update knowledge base based on empirical feedback"
        }
