"""
Shader Library - Quantum Shader Library Management
================================================

The Shader Library provides management for quantum shader libraries,
including registration, discovery, and marketplace functionality.

This is the GOD-TIER shader library system that enables
community-driven quantum shader development and sharing.
"""

import time
import logging
import numpy as np
import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import hashlib
import requests

logger = logging.getLogger(__name__)

class LibraryType(Enum):
    """Types of shader libraries."""
    OFFICIAL = "official"
    COMMUNITY = "community"
    RESEARCH = "research"
    COMMERCIAL = "commercial"
    CUSTOM = "custom"

class ShaderLibrary:
    """
    Shader Library for Quantum Shader Management.
    
    This manages quantum shader libraries with registration,
    discovery, and marketplace functionality.
    """
    
    def __init__(self):
        """Initialize the shader library."""
        self.libraries: Dict[str, Dict[str, Any]] = {}
        self.shader_registry: Dict[str, List[str]] = defaultdict(list)
        self.library_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Library statistics
        self.library_stats = {
            'total_libraries': 0,
            'total_shaders': 0,
            'community_shaders': 0,
            'official_shaders': 0,
            'download_count': 0,
            'average_rating': 0.0
        }
        
        logger.info("🎨 Shader Library initialized - Quantum shader management active")
    
    def register_library(self, library_id: str, name: str, description: str,
                        library_type: LibraryType, author: str = "",
                        version: str = "1.0.0") -> bool:
        """Register a new shader library."""
        library_info = {
            'library_id': library_id,
            'name': name,
            'description': description,
            'type': library_type.value,
            'author': author,
            'version': version,
            'created_at': time.time(),
            'updated_at': time.time(),
            'shader_count': 0,
            'download_count': 0,
            'rating': 0.0
        }
        
        self.libraries[library_id] = library_info
        self.library_metadata[library_id] = library_info
        
        self.library_stats['total_libraries'] += 1
        
        logger.info(f"🎨 Registered shader library: {name} (ID: {library_id})")
        return True
    
    def add_shader_to_library(self, library_id: str, shader_id: str) -> bool:
        """Add a shader to a library."""
        if library_id not in self.libraries:
            return False
        
        self.shader_registry[library_id].append(shader_id)
        self.libraries[library_id]['shader_count'] += 1
        self.library_stats['total_shaders'] += 1
        
        logger.info(f"🎨 Added shader {shader_id} to library {library_id}")
        return True
    
    def get_library(self, library_id: str) -> Optional[Dict[str, Any]]:
        """Get library information."""
        return self.libraries.get(library_id)
    
    def list_libraries(self, library_type: Optional[LibraryType] = None) -> List[Dict[str, Any]]:
        """List libraries, optionally filtered by type."""
        if library_type is None:
            return list(self.libraries.values())
        
        return [lib for lib in self.libraries.values() if lib['type'] == library_type.value]
    
    def search_shaders(self, query: str, library_id: Optional[str] = None) -> List[str]:
        """Search for shaders in libraries."""
        results = []
        
        libraries_to_search = [library_id] if library_id else list(self.libraries.keys())
        
        for lib_id in libraries_to_search:
            if lib_id in self.shader_registry:
                for shader_id in self.shader_registry[lib_id]:
                    # Simple search implementation
                    if query.lower() in shader_id.lower():
                        results.append(shader_id)
        
        return results
    
    def get_library_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        return {
            'library_stats': self.library_stats,
            'libraries_by_type': {
                lib_type.value: sum(1 for lib in self.libraries.values() if lib['type'] == lib_type.value)
                for lib_type in LibraryType
            },
            'total_libraries': len(self.libraries),
            'total_shaders': sum(len(shaders) for shaders in self.shader_registry.values())
        }

class ShaderRegistry:
    """
    Shader Registry for Quantum Shader Registration.
    
    This provides centralized registration and discovery
    of quantum shaders across the ecosystem.
    """
    
    def __init__(self):
        """Initialize the shader registry."""
        self.registered_shaders: Dict[str, Dict[str, Any]] = {}
        self.shader_categories: Dict[str, List[str]] = defaultdict(list)
        self.shader_tags: Dict[str, List[str]] = defaultdict(list)
        
        # Registry statistics
        self.registry_stats = {
            'total_registrations': 0,
            'active_shaders': 0,
            'deprecated_shaders': 0,
            'average_rating': 0.0,
            'total_downloads': 0
        }
        
        logger.info("🎨 Shader Registry initialized - Quantum shader registration active")
    
    def register_shader(self, shader_id: str, name: str, description: str,
                       category: str, tags: List[str] = None,
                       author: str = "", version: str = "1.0.0") -> bool:
        """Register a quantum shader."""
        shader_info = {
            'shader_id': shader_id,
            'name': name,
            'description': description,
            'category': category,
            'tags': tags or [],
            'author': author,
            'version': version,
            'registered_at': time.time(),
            'download_count': 0,
            'rating': 0.0,
            'status': 'active'
        }
        
        self.registered_shaders[shader_id] = shader_info
        self.shader_categories[category].append(shader_id)
        
        for tag in shader_info['tags']:
            self.shader_tags[tag].append(shader_id)
        
        self.registry_stats['total_registrations'] += 1
        self.registry_stats['active_shaders'] += 1
        
        logger.info(f"🎨 Registered shader: {name} (ID: {shader_id})")
        return True
    
    def get_shader(self, shader_id: str) -> Optional[Dict[str, Any]]:
        """Get shader information."""
        return self.registered_shaders.get(shader_id)
    
    def list_shaders_by_category(self, category: str) -> List[Dict[str, Any]]:
        """List shaders by category."""
        shader_ids = self.shader_categories.get(category, [])
        return [self.registered_shaders[shader_id] for shader_id in shader_ids 
                if shader_id in self.registered_shaders]
    
    def list_shaders_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """List shaders by tag."""
        shader_ids = self.shader_tags.get(tag, [])
        return [self.registered_shaders[shader_id] for shader_id in shader_ids 
                if shader_id in self.registered_shaders]
    
    def search_shaders(self, query: str) -> List[Dict[str, Any]]:
        """Search for shaders."""
        results = []
        
        for shader_id, shader_info in self.registered_shaders.items():
            if (query.lower() in shader_info['name'].lower() or 
                query.lower() in shader_info['description'].lower() or
                query.lower() in shader_info['category'].lower()):
                results.append(shader_info)
        
        return results
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            'registry_stats': self.registry_stats,
            'categories': list(self.shader_categories.keys()),
            'tags': list(self.shader_tags.keys()),
            'total_shaders': len(self.registered_shaders)
        }

class ShaderMarketplace:
    """
    Shader Marketplace for Quantum Shader Commerce.
    
    This provides marketplace functionality for buying,
    selling, and sharing quantum shaders.
    """
    
    def __init__(self):
        """Initialize the shader marketplace."""
        self.marketplace_items: Dict[str, Dict[str, Any]] = {}
        self.transactions: List[Dict[str, Any]] = []
        self.user_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Marketplace statistics
        self.marketplace_stats = {
            'total_items': 0,
            'total_transactions': 0,
            'total_revenue': 0.0,
            'average_rating': 0.0,
            'active_sellers': 0,
            'active_buyers': 0
        }
        
        logger.info("🎨 Shader Marketplace initialized - Quantum shader commerce active")
    
    def list_shader(self, shader_id: str, price: float, description: str = "",
                   seller: str = "", category: str = "quantum") -> bool:
        """List a shader for sale."""
        item_id = f"item_{int(time.time() * 1000)}"
        
        marketplace_item = {
            'item_id': item_id,
            'shader_id': shader_id,
            'price': price,
            'description': description,
            'seller': seller,
            'category': category,
            'listed_at': time.time(),
            'status': 'available',
            'download_count': 0,
            'rating': 0.0
        }
        
        self.marketplace_items[item_id] = marketplace_item
        self.marketplace_stats['total_items'] += 1
        
        logger.info(f"🎨 Listed shader for sale: {shader_id} at ${price}")
        return True
    
    def purchase_shader(self, item_id: str, buyer: str, payment_method: str = "credit") -> bool:
        """Purchase a shader."""
        if item_id not in self.marketplace_items:
            return False
        
        item = self.marketplace_items[item_id]
        
        # Create transaction
        transaction = {
            'transaction_id': f"txn_{int(time.time() * 1000)}",
            'item_id': item_id,
            'shader_id': item['shader_id'],
            'buyer': buyer,
            'seller': item['seller'],
            'price': item['price'],
            'payment_method': payment_method,
            'timestamp': time.time(),
            'status': 'completed'
        }
        
        self.transactions.append(transaction)
        self.marketplace_stats['total_transactions'] += 1
        self.marketplace_stats['total_revenue'] += item['price']
        
        # Update item status
        item['status'] = 'sold'
        item['download_count'] += 1
        
        logger.info(f"🎨 Shader purchased: {item['shader_id']} by {buyer}")
        return True
    
    def rate_shader(self, shader_id: str, user: str, rating: float, 
                   review: str = "") -> bool:
        """Rate a shader."""
        if not (0.0 <= rating <= 5.0):
            return False
        
        self.user_ratings[shader_id][user] = rating
        
        # Update average rating
        ratings = list(self.user_ratings[shader_id].values())
        avg_rating = sum(ratings) / len(ratings)
        
        # Update marketplace item rating
        for item in self.marketplace_items.values():
            if item['shader_id'] == shader_id:
                item['rating'] = avg_rating
        
        logger.info(f"🎨 Shader rated: {shader_id} by {user} ({rating}/5.0)")
        return True
    
    def get_marketplace_statistics(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        return {
            'marketplace_stats': self.marketplace_stats,
            'total_items': len(self.marketplace_items),
            'total_transactions': len(self.transactions),
            'categories': list(set(item['category'] for item in self.marketplace_items.values()))
        }
