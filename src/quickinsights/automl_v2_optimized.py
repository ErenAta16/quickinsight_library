"""
QuickInsights - Optimized AutoML v2 Module

This module provides optimized automated machine learning capabilities with:
- Efficient model selection
- Automated hyperparameter tuning
- Performance optimization
- Memory management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from functools import lru_cache
import time
import gc

warnings.filterwarnings("ignore")


# Optimized lazy loading with caching
@lru_cache(maxsize=32)
def _get_ml_libraries_cached():
    """Get ML libraries with cached lazy loading."""
    try:
        from ._imports import (
            get_sklearn_utils,
            get_lightgbm_utils,
            get_xgboost_utils,
            get_shap_utils,
        )

        sklearn_utils = get_sklearn_utils()
        lightgbm_utils = get_lightgbm_utils()
        xgboost_utils = get_xgboost_utils()
        shap_utils = get_shap_utils()

        return sklearn_utils, lightgbm_utils, xgboost_utils, shap_utils
    except ImportError:
        return None, None, None, None


def _get_ml_libraries():
    """Get ML libraries with lazy loading."""
    return _get_ml_libraries_cached()


# Check availability without printing
def _check_ml_availability():
    """Check ML library availability silently."""
    sklearn_utils, lightgbm_utils, xgboost_utils, shap_utils = _get_ml_libraries()

    sklearn_available = sklearn_utils is not None and sklearn_utils.get(
        "available", False
    )
    lightgbm_available = lightgbm_utils is not None and lightgbm_utils.get(
        "available", False
    )
    xgboost_available = xgboost_utils is not None and xgboost_utils.get(
        "available", False
    )
    shap_available = shap_utils is not None and shap_utils.get("available", False)

    return sklearn_available, lightgbm_available, xgboost_available, shap_available


# Global availability flags
(
    SKLEARN_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    XGBOOST_AVAILABLE,
    SHAP_AVAILABLE,
) = _check_ml_availability()


class AutoMLOptimized:
    """
    Optimized AutoML engine with performance improvements:
    - Efficient model selection
    - Automated hyperparameter tuning
    - Memory optimization
    - Caching strategies
    """

    def __init__(self, enable_caching: bool = True, max_memory_mb: int = 1024):
        """
        Initialize Optimized AutoML engine

        Parameters
        ----------
        enable_caching : bool, default True
            Enable result caching for better performance
        max_memory_mb : int, default 1024
            Maximum memory usage in MB
        """
        self.enable_caching = enable_caching
        self.max_memory_mb = max_memory_mb

        # Initialize caches
        self._model_cache = {}
        self._performance_cache = {}
        self._feature_cache = {}

        # Performance tracking
        self._training_times = {}
        self._memory_usage = {}

        # Available models based on library availability
        self._available_models = self._get_available_models()

    def _get_available_models(self) -> Dict[str, bool]:
        """Get available models based on library availability"""
        models = {
            "sklearn": SKLEARN_AVAILABLE,
            "lightgbm": LIGHTGBM_AVAILABLE,
            "xgboost": XGBOOST_AVAILABLE,
            "shap": SHAP_AVAILABLE,
        }
        return models

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < self.max_memory_mb
        except ImportError:
            return True  # If psutil not available, assume OK

    def _prepare_data_efficient(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Efficient data preparation with memory management"""
        start_time = time.time()

        # Convert to numpy arrays for efficiency
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        if isinstance(y, pd.Series):
            y_array = y.values
        elif isinstance(y, str):
            # If y is a column name, extract it from X
            if isinstance(X, pd.DataFrame):
                y_array = X[y].values
                X_array = X.drop(columns=[y]).values
            else:
                raise ValueError("If y is a string, X must be a DataFrame")
        else:
            y_array = y

        # Handle missing values efficiently
        if X_array.dtype.kind in "fc":  # float or complex
            if np.isnan(X_array).any():
                # Use median for numeric columns
                X_array = np.where(
                    np.isnan(X_array), np.nanmedian(X_array, axis=0), X_array
                )

        if y_array.dtype.kind in "fc":  # float or complex
            if np.isnan(y_array).any():
                # Use mode for target
                y_array = np.where(np.isnan(y_array), np.nanmedian(y_array), y_array)

        # Memory cleanup
        gc.collect()

        if self.enable_caching:
            print(
                f"⚡ Data preparation completed in {time.time() - start_time:.4f} seconds"
            )

        return X_array, y_array

    def select_best_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "auto",
        max_models: int = 5,
    ) -> Dict[str, Any]:
        """
        Select best model based on task type and data characteristics

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        task_type : str, default 'auto'
            Task type: 'classification', 'regression', or 'auto'
        max_models : int, default 5
            Maximum number of models to evaluate

        Returns
        -------
        Dict[str, Any]
            Best model information and performance metrics
        """
        cache_key = f"best_model_{task_type}_{max_models}_{hash(str(X.shape))}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        start_time = time.time()

        # Auto-detect task type
        if task_type == "auto":
            if y.dtype in ["object", "category"] or len(y.unique()) < 10:
                task_type = "classification"
            else:
                task_type = "regression"

        # Prepare data efficiently
        X_array, y_array = self._prepare_data_efficient(X, y)

        # Select models based on task type and availability
        models_to_test = self._get_models_for_task(task_type, max_models)

        best_model = None
        best_score = -np.inf if task_type == "regression" else 0
        best_model_name = None

        for model_name, model_config in models_to_test.items():
            try:
                # Check memory usage
                if not self._check_memory_usage():
                    print(f"⚠️ Memory limit reached, stopping at {model_name}")
                    break

                # Train and evaluate model
                model, score = self._train_and_evaluate_model(
                    model_name, model_config, X_array, y_array, task_type
                )

                # Update best model
                if self._is_better_score(score, best_score, task_type):
                    best_score = score
                    best_model = model
                    best_model_name = model_name

                # Cache individual model performance
                self._performance_cache[model_name] = {
                    "score": score,
                    "training_time": self._training_times.get(model_name, 0),
                    "memory_usage": self._memory_usage.get(model_name, 0),
                }

            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {str(e)}")
                continue

        # Compile results
        result = {
            "best_model_name": best_model_name,
            "best_model": best_model,
            "best_score": best_score,
            "task_type": task_type,
            "models_tested": list(models_to_test.keys()),
            "performance_summary": self._performance_cache,
            "execution_time": time.time() - start_time,
            "optimization_level": "high",
        }

        # Cache results
        if self.enable_caching:
            self._model_cache[cache_key] = result

        return result

    def _get_models_for_task(self, task_type: str, max_models: int) -> Dict[str, Dict]:
        """Get models to test for specific task type"""
        models = {}

        if task_type == "classification":
            if SKLEARN_AVAILABLE:
                models.update(
                    {
                        "logistic_regression": {
                            "type": "sklearn",
                            "params": {"max_iter": 1000},
                        },
                        "random_forest": {
                            "type": "sklearn",
                            "params": {"n_estimators": 100},
                        },
                        "gradient_boosting": {
                            "type": "sklearn",
                            "params": {"n_estimators": 100},
                        },
                    }
                )

            if LIGHTGBM_AVAILABLE:
                models["lightgbm"] = {
                    "type": "lightgbm",
                    "params": {"n_estimators": 100},
                }

            if XGBOOST_AVAILABLE:
                models["xgboost"] = {"type": "xgboost", "params": {"n_estimators": 100}}

        elif task_type == "regression":
            if SKLEARN_AVAILABLE:
                models.update(
                    {
                        "linear_regression": {"type": "sklearn", "params": {}},
                        "random_forest_regressor": {
                            "type": "sklearn",
                            "params": {"n_estimators": 100},
                        },
                        "gradient_boosting_regressor": {
                            "type": "sklearn",
                            "params": {"n_estimators": 100},
                        },
                    }
                )

            if LIGHTGBM_AVAILABLE:
                models["lightgbm_regressor"] = {
                    "type": "lightgbm",
                    "params": {"n_estimators": 100},
                }

            if XGBOOST_AVAILABLE:
                models["xgboost_regressor"] = {
                    "type": "xgboost",
                    "params": {"n_estimators": 100},
                }

        # Limit number of models
        return dict(list(models.items())[:max_models])

    def _train_and_evaluate_model(
        self,
        model_name: str,
        model_config: Dict,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
    ) -> Tuple[Any, float]:
        """Train and evaluate a single model"""
        start_time = time.time()

        # Get model instance
        model = self._get_model_instance(model_name, model_config, task_type)

        # Train model
        model.fit(X, y)

        # Evaluate model
        if task_type == "classification":
            score = model.score(X, y)
        else:  # regression
            score = model.score(X, y)

        # Record performance
        training_time = time.time() - start_time
        self._training_times[model_name] = training_time

        # Memory usage tracking
        if self._check_memory_usage():
            try:
                import psutil

                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self._memory_usage[model_name] = memory_mb
            except ImportError:
                self._memory_usage[model_name] = 0

        return model, score

    def _get_model_instance(
        self, model_name: str, model_config: Dict, task_type: str
    ) -> Any:
        """Get model instance based on configuration"""
        model_type = model_config["type"]
        params = model_config.get("params", {})

        if model_type == "sklearn":
            sklearn_utils = _get_ml_libraries()[0]

            if task_type == "classification":
                if "logistic" in model_name:
                    return sklearn_utils["LogisticRegression"](**params)
                elif "random_forest" in model_name:
                    return sklearn_utils["RandomForestClassifier"](**params)
                elif "gradient_boosting" in model_name:
                    return sklearn_utils["GradientBoostingClassifier"](**params)
            else:  # regression
                if "linear" in model_name:
                    return sklearn_utils["LinearRegression"](**params)
                elif "random_forest" in model_name:
                    return sklearn_utils["RandomForestRegressor"](**params)
                elif "gradient_boosting" in model_name:
                    return sklearn_utils["GradientBoostingRegressor"](**params)

        elif model_type == "lightgbm":
            lightgbm_utils = _get_ml_libraries()[1]
            if task_type == "classification":
                return lightgbm_utils["LGBMClassifier"](**params)
            else:
                return lightgbm_utils["LGBMRegressor"](**params)

        elif model_type == "xgboost":
            xgboost_utils = _get_ml_libraries()[2]
            if task_type == "classification":
                return xgboost_utils["XGBClassifier"](**params)
            else:
                return xgboost_utils["XGBRegressor"](**params)

        # Fallback to sklearn
        sklearn_utils = _get_ml_libraries()[0]
        if task_type == "classification":
            return sklearn_utils["RandomForestClassifier"](**params)
        else:
            return sklearn_utils["RandomForestRegressor"](**params)

    def _is_better_score(
        self, new_score: float, current_best: float, task_type: str
    ) -> bool:
        """Check if new score is better than current best"""
        if task_type == "regression":
            # For regression, higher R² is better
            return new_score > current_best
        else:
            # For classification, higher accuracy is better
            return new_score > current_best

    def get_feature_importance(
        self, model: Any, feature_names: List[str] = None
    ) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

        try:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_)
            else:
                return {}

            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importance))

            # Sort by importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            return sorted_importance

        except Exception as e:
            print(f"⚠️ Could not extract feature importance: {str(e)}")
            return {}

    def clear_cache(self):
        """Clear all cached results for memory management"""
        self._model_cache.clear()
        self._performance_cache.clear()
        self._feature_cache.clear()
        self._training_times.clear()
        self._memory_usage.clear()
        gc.collect()
        print("🧹 AutoML cache cleared for memory optimization")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and optimization metrics"""
        return {
            "cache_size": {
                "models": len(self._model_cache),
                "performance": len(self._performance_cache),
                "features": len(self._feature_cache),
            },
            "training_times": self._training_times,
            "memory_usage": self._memory_usage,
            "available_libraries": self._available_models,
            "optimization_features": {
                "lazy_loading": True,
                "caching": self.enable_caching,
                "memory_management": True,
                "efficient_data_processing": True,
            },
        }
