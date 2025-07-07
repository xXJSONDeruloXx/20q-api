"""
Demographic segmentation and cohort-based weight management.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
import logging

from .io import WeightIO

logger = logging.getLogger(__name__)


class CohortType(Enum):
    """Types of demographic cohorts."""
    
    GEOGRAPHIC = "geographic"
    AGE = "age"
    CULTURAL = "cultural"
    LINGUISTIC = "linguistic"
    CUSTOM = "custom"


@dataclass
class Cohort:
    """Represents a demographic cohort."""
    
    id: str
    name: str
    type: CohortType
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DemographicManager:
    """Manages demographic cohorts and their associated weight matrices."""
    
    def __init__(self, cohorts_dir: Optional[Path] = None):
        """
        Initialize demographic manager.
        
        Args:
            cohorts_dir: Directory containing cohort weight files
        """
        self.cohorts: Dict[str, Cohort] = {}
        self.cohorts_dir = cohorts_dir or Path("data/priors/cohort_weights")
        self.cohorts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load available cohorts
        self._discover_cohorts()
    
    def _discover_cohorts(self) -> None:
        """Discover available cohorts from directory structure."""
        if not self.cohorts_dir.exists():
            return
            
        # Look for weight files and infer cohorts
        for weight_file in self.cohorts_dir.glob("*.bin"):
            cohort_id = weight_file.stem
            
            # Try to load metadata
            metadata_file = weight_file.with_suffix(".json")
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {cohort_id}: {e}")
            
            # Create cohort
            cohort = Cohort(
                id=cohort_id,
                name=metadata.get('name', cohort_id.replace('_', ' ').title()),
                type=CohortType(metadata.get('type', 'custom')),
                description=metadata.get('description', ''),
                metadata=metadata
            )
            
            self.cohorts[cohort_id] = cohort
            logger.info(f"Discovered cohort: {cohort_id}")
    
    def add_cohort(self, cohort: Cohort) -> None:
        """Add a new cohort."""
        self.cohorts[cohort.id] = cohort
        
        # Save metadata
        metadata_file = self.cohorts_dir / f"{cohort.id}.json"
        self._save_cohort_metadata(cohort, metadata_file)
    
    def remove_cohort(self, cohort_id: str) -> None:
        """Remove a cohort and its associated files."""
        if cohort_id in self.cohorts:
            del self.cohorts[cohort_id]
            
            # Remove files
            weight_file = self.cohorts_dir / f"{cohort_id}.bin"
            metadata_file = self.cohorts_dir / f"{cohort_id}.json"
            
            if weight_file.exists():
                weight_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
    
    def get_cohort(self, cohort_id: str) -> Optional[Cohort]:
        """Get a cohort by ID."""
        return self.cohorts.get(cohort_id)
    
    def list_cohorts(self) -> List[Cohort]:
        """Get list of all available cohorts."""
        return list(self.cohorts.values())
    
    def save_cohort_weights(
        self, 
        cohort_id: str, 
        weights: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save weight matrix for a cohort.
        
        Args:
            cohort_id: ID of the cohort
            weights: Weight matrix to save
            metadata: Additional metadata
        """
        weight_file = self.cohorts_dir / f"{cohort_id}.bin"
        WeightIO.save_weights(weights, weight_file)
        
        # Update or create cohort
        if cohort_id not in self.cohorts:
            cohort = Cohort(
                id=cohort_id,
                name=cohort_id.replace('_', ' ').title(),
                type=CohortType.CUSTOM,
                metadata=metadata or {}
            )
            self.cohorts[cohort_id] = cohort
        
        # Save metadata
        metadata_file = self.cohorts_dir / f"{cohort_id}.json"
        self._save_cohort_metadata(self.cohorts[cohort_id], metadata_file)
        
        logger.info(f"Saved weights for cohort: {cohort_id}")
    
    def load_cohort_weights(
        self, 
        cohort: Cohort, 
        n_objects: int, 
        n_questions: int
    ) -> Optional[np.ndarray]:
        """
        Load weight matrix for a cohort.
        
        Args:
            cohort: Cohort to load weights for
            n_objects: Expected number of objects
            n_questions: Expected number of questions
            
        Returns:
            Weight matrix or None if not found
        """
        weight_file = self.cohorts_dir / f"{cohort.id}.bin"
        
        if not weight_file.exists():
            logger.warning(f"Weight file not found for cohort: {cohort.id}")
            return None
            
        try:
            weights = WeightIO.load_weights(weight_file)
            
            # Validate dimensions
            if weights.shape != (n_objects, n_questions):
                logger.warning(
                    f"Weight matrix shape mismatch for cohort {cohort.id}: "
                    f"expected {(n_objects, n_questions)}, got {weights.shape}"
                )
                return None
                
            logger.info(f"Loaded weights for cohort: {cohort.id}")
            return weights
            
        except Exception as e:
            logger.error(f"Failed to load weights for cohort {cohort.id}: {e}")
            return None
    
    def _save_cohort_metadata(self, cohort: Cohort, filepath: Path) -> None:
        """Save cohort metadata to JSON file."""
        metadata = {
            'id': cohort.id,
            'name': cohort.name,
            'type': cohort.type.value,
            'description': cohort.description,
            **cohort.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def infer_cohort(self, user_profile: Dict[str, Any]) -> Optional[Cohort]:
        """
        Infer the best cohort for a user based on their profile.
        
        Args:
            user_profile: User profile data (location, age, language, etc.)
            
        Returns:
            Best matching cohort or None
        """
        # Simple heuristic-based matching
        # In a real implementation, this would use more sophisticated ML
        
        scores = {}
        
        for cohort in self.cohorts.values():
            score = 0.0
            
            # Geographic matching
            if cohort.type == CohortType.GEOGRAPHIC:
                user_country = user_profile.get('country', '').lower()
                user_region = user_profile.get('region', '').lower()
                
                cohort_countries = cohort.metadata.get('countries', [])
                cohort_regions = cohort.metadata.get('regions', [])
                
                if user_country in [c.lower() for c in cohort_countries]:
                    score += 2.0
                if user_region in [r.lower() for r in cohort_regions]:
                    score += 1.0
            
            # Age matching
            elif cohort.type == CohortType.AGE:
                user_age = user_profile.get('age')
                if user_age is not None:
                    age_min = cohort.metadata.get('age_min', 0)
                    age_max = cohort.metadata.get('age_max', 100)
                    
                    if age_min <= user_age <= age_max:
                        score += 1.5
            
            # Language matching
            elif cohort.type == CohortType.LINGUISTIC:
                user_language = user_profile.get('language', '').lower()
                cohort_languages = cohort.metadata.get('languages', [])
                
                if user_language in [l.lower() for l in cohort_languages]:
                    score += 1.0
            
            scores[cohort.id] = score
        
        # Return best matching cohort
        if scores:
            best_cohort_id = max(scores, key=scores.get)
            if scores[best_cohort_id] > 0:
                return self.cohorts[best_cohort_id]
        
        return None
    
    def create_default_cohorts(self) -> None:
        """Create some default cohorts for common demographics."""
        
        # US cohort
        us_cohort = Cohort(
            id="us",
            name="United States",
            type=CohortType.GEOGRAPHIC,
            description="US-based demographic cohort",
            metadata={
                'countries': ['united states', 'usa', 'us'],
                'regions': ['north america'],
                'languages': ['english']
            }
        )
        self.add_cohort(us_cohort)
        
        # European cohort
        eu_cohort = Cohort(
            id="eu",
            name="Europe",
            type=CohortType.GEOGRAPHIC,
            description="European demographic cohort",
            metadata={
                'countries': ['germany', 'france', 'italy', 'spain', 'uk', 'netherlands'],
                'regions': ['europe'],
                'languages': ['english', 'german', 'french', 'italian', 'spanish']
            }
        )
        self.add_cohort(eu_cohort)
        
        # Asian cohort
        asia_cohort = Cohort(
            id="asia",
            name="Asia",
            type=CohortType.GEOGRAPHIC,
            description="Asian demographic cohort",
            metadata={
                'countries': ['japan', 'china', 'korea', 'india', 'singapore'],
                'regions': ['asia'],
                'languages': ['english', 'japanese', 'chinese', 'korean', 'hindi']
            }
        )
        self.add_cohort(asia_cohort)
        
        # Young adults cohort
        young_cohort = Cohort(
            id="young_adults",
            name="Young Adults",
            type=CohortType.AGE,
            description="Young adults (18-35)",
            metadata={
                'age_min': 18,
                'age_max': 35
            }
        )
        self.add_cohort(young_cohort)
        
        # Middle age cohort
        middle_cohort = Cohort(
            id="middle_age",
            name="Middle Age",
            type=CohortType.AGE,
            description="Middle-aged adults (36-55)",
            metadata={
                'age_min': 36,
                'age_max': 55
            }
        )
        self.add_cohort(middle_cohort)
        
        logger.info("Created default cohorts")
    
    def get_cohort_stats(self, cohort_id: str) -> Dict[str, Any]:
        """Get statistics for a cohort."""
        cohort = self.get_cohort(cohort_id)
        if not cohort:
            return {}
            
        weight_file = self.cohorts_dir / f"{cohort_id}.bin"
        
        stats = {
            'id': cohort.id,
            'name': cohort.name,
            'type': cohort.type.value,
            'description': cohort.description,
            'has_weights': weight_file.exists(),
            'weight_file_size': weight_file.stat().st_size if weight_file.exists() else 0,
            'metadata': cohort.metadata
        }
        
        return stats
    
    def merge_cohorts(
        self, 
        cohort_ids: List[str], 
        new_cohort_id: str,
        weights: List[float] = None
    ) -> Optional[Cohort]:
        """
        Merge multiple cohorts into a new one.
        
        Args:
            cohort_ids: List of cohort IDs to merge
            new_cohort_id: ID for the new merged cohort
            weights: Optional weights for averaging (default: equal weights)
            
        Returns:
            New merged cohort or None if merge failed
        """
        if not cohort_ids:
            return None
            
        # Validate all cohorts exist
        cohorts = []
        for cohort_id in cohort_ids:
            cohort = self.get_cohort(cohort_id)
            if not cohort:
                logger.error(f"Cohort not found: {cohort_id}")
                return None
            cohorts.append(cohort)
        
        # Default equal weights
        if weights is None:
            weights = [1.0 / len(cohorts)] * len(cohorts)
        elif len(weights) != len(cohorts):
            logger.error("Number of weights must match number of cohorts")
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Create merged cohort
        merged_cohort = Cohort(
            id=new_cohort_id,
            name=f"Merged: {', '.join(c.name for c in cohorts)}",
            type=CohortType.CUSTOM,
            description=f"Merged from cohorts: {', '.join(cohort_ids)}",
            metadata={
                'source_cohorts': cohort_ids,
                'merge_weights': weights
            }
        )
        
        self.add_cohort(merged_cohort)
        logger.info(f"Created merged cohort: {new_cohort_id}")
        
        return merged_cohort
