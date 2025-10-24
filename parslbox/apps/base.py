"""
Base Application Class for ParslBox

This module defines the abstract base class for application implementations,
providing a consistent interface for all supported applications.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class AppBase(ABC):
    """
    Abstract base class for ParslBox applications.
    
    Each application (LAMMPS, VASP, Python, etc.) should inherit from this class
    and implement the required methods and class attributes.
    """
    
    # App configuration (must be defined in subclasses)
    INPUT_REQUIRED: bool
    DFLT_INPUT: str | None
    
    @abstractmethod
    def preprocess(self, job_id: int, job_path: Path, db_path: Path, app_config: dict, config_name: str):
        """
        Preprocessing before job execution.
        
        This method is called before the main parsl_app execution.
        Default implementation should do nothing for most apps.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            app_config (dict): Application configuration from YAML
            config_name (str): Name of the system configuration (e.g., 'polaris')
        """
        pass
    
    @abstractmethod
    def parsl_app(self, job_id: int, job_path: Path, db_path: Path, ngpus: int, 
                  app_config: dict, config_name: str, in_file: str, mpi_opts: str, stdout: str, stderr: str):
        """
        Main Parsl app function for executing the application.
        
        This method should be decorated with @bash_app or @python_app.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            ngpus (int): Number of GPUs allocated for this job
            app_config (dict): Application configuration from YAML
            config_name (str): Name of the system configuration (e.g., 'polaris')
            in_file (str): Input filename for the job
            mpi_opts (str): Additional MPI options to append to the MPI command
            stdout (str): Standard output file path
            stderr (str): Standard error file path
        """
        pass
    
    @abstractmethod
    def check_success(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Check if the job completed successfully and return final status.
        
        This method is called after the parsl_app completes to determine
        the final job status based on output files or other criteria.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            
        Returns:
            str: Final job status ('Done', 'Failed', etc.)
        """
        pass
    
    @abstractmethod
    def postprocess(self, job_id: int, job_path: Path, db_path: Path):
        """
        Perform any post-processing after job completion.
        
        This method is called after check_success to perform any
        cleanup or additional processing tasks.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
        """
        pass
