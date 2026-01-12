"""
Pydantic schema for validating the application's configuration file.

This module defines the expected structure, data types, and constraints
for the `config/default.yaml` file. It helps prevent silent failures
from typos or invalid values in the configuration.
"""
from typing import List
from pydantic import BaseModel, Field, conlist, confloat


class ModelConfig(BaseModel):
    """Schema for the 'model' section of the config."""
    type: str
    embedding_model: str
    max_seq_len: int = Field(..., gt=0, description="Max sequence length must be positive.")
    cnn_filters: int = Field(..., gt=0, description="Number of CNN filters must be positive.")
    lstm_units: int = Field(..., gt=0, description="Number of LSTM units must be positive.")
    dropout: confloat(ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    """Schema for the 'training' section of the config."""
    batch_size: int = Field(..., gt=0)
    epochs: int = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0)


class AdversarialConfig(BaseModel):
    """Schema for the 'adversarial' section of the config."""
    enabled: bool
    methods: List[str]


class AppConfig(BaseModel):
    """Root schema for the entire configuration."""
    project_name: str
    languages: conlist(str, min_length=1) = Field(..., description="List of languages must contain at least one language.")
    model: ModelConfig
    training: TrainingConfig
    adversarial: AdversarialConfigag