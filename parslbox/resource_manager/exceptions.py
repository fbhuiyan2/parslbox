"""
Resource Manager Exceptions

Custom exceptions for resource management operations.
"""


class ResourceManagerError(Exception):
    """Base exception for resource manager errors."""
    pass


class InsufficientResources(ResourceManagerError):
    """Raised when there are not enough resources available for a job request."""
    
    def __init__(self, message: str, requested: dict = None, available: dict = None):
        super().__init__(message)
        self.requested = requested or {}
        self.available = available or {}
    
    def __str__(self):
        base_msg = super().__str__()
        if self.requested and self.available:
            return f"{base_msg} (requested: {self.requested}, available: {self.available})"
        return base_msg


class InvalidResourceSpec(ResourceManagerError):
    """Raised when a resource specification is invalid."""
    pass


class JobNotFound(ResourceManagerError):
    """Raised when trying to free resources for a job that doesn't exist."""
    pass


class NodeNotFound(ResourceManagerError):
    """Raised when a specified node is not found in the resource manager."""
    pass
