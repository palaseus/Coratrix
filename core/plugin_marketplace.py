"""
Plugin Marketplace for Coratrix 4.0

This module implements a plugin marketplace within the Coratrix ecosystem,
enabling community-contributed plugins with a review system for quality control.
"""

import json
import os
import sys
import tempfile
import shutil
import hashlib
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod
import requests
from pathlib import Path
import time
from datetime import datetime, timedelta

# Web framework imports
try:
    from flask import Flask, render_template, request, jsonify, send_file, session
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    render_template = None
    request = None
    jsonify = None
    send_file = None
    session = None
    CORS = None

# Database imports
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    sqlite3 = None

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Plugin status in marketplace."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    UNDER_REVIEW = "under_review"


class ReviewStatus(Enum):
    """Review status for plugins."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class PluginCategory(Enum):
    """Plugin categories."""
    QUANTUM_GATES = "quantum_gates"
    COMPILER_PASSES = "compiler_passes"
    BACKENDS = "backends"
    VISUALIZATIONS = "visualizations"
    NOISE_MODELS = "noise_models"
    OPTIMIZERS = "optimizers"
    ALGORITHMS = "algorithms"
    UTILITIES = "utilities"


@dataclass
class PluginReview:
    """Plugin review information."""
    reviewer_id: str
    reviewer_name: str
    review_date: datetime
    status: ReviewStatus
    score: int  # 1-5 stars
    comments: str
    technical_review: str
    security_review: str
    performance_review: str
    documentation_review: str


@dataclass
class PluginStats:
    """Plugin statistics."""
    downloads: int
    rating: float
    reviews_count: int
    last_updated: datetime
    compatibility_score: float
    performance_score: float


@dataclass
class MarketplacePlugin:
    """Plugin in the marketplace."""
    id: str
    name: str
    version: str
    description: str
    author: str
    category: PluginCategory
    status: PluginStatus
    tags: List[str]
    dependencies: List[str]
    license: str
    homepage: Optional[str]
    repository: Optional[str]
    documentation: Optional[str]
    created_date: datetime
    updated_date: datetime
    reviews: List[PluginReview]
    stats: PluginStats
    download_url: str
    checksum: str
    file_size: int
    compatibility: Dict[str, str]  # version -> compatibility


class PluginMarketplace:
    """
    Main plugin marketplace class.
    """
    
    def __init__(self, db_path: str = "marketplace.db", 
                 plugins_dir: str = "marketplace_plugins"):
        """
        Initialize plugin marketplace.
        
        Args:
            db_path: Path to SQLite database
            plugins_dir: Directory for plugin files
        """
        self.db_path = db_path
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize web server if Flask is available
        if FLASK_AVAILABLE:
            self.app = self._create_web_app()
        else:
            self.app = None
            logger.warning("Flask not available, web interface disabled")
    
    def _init_database(self):
        """Initialize SQLite database."""
        if not SQLITE_AVAILABLE:
            logger.warning("SQLite not available, using file-based storage")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plugins (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                author TEXT,
                category TEXT,
                status TEXT,
                tags TEXT,
                dependencies TEXT,
                license TEXT,
                homepage TEXT,
                repository TEXT,
                documentation TEXT,
                created_date TEXT,
                updated_date TEXT,
                download_url TEXT,
                checksum TEXT,
                file_size INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plugin_id TEXT,
                reviewer_id TEXT,
                reviewer_name TEXT,
                review_date TEXT,
                status TEXT,
                score INTEGER,
                comments TEXT,
                technical_review TEXT,
                security_review TEXT,
                performance_review TEXT,
                documentation_review TEXT,
                FOREIGN KEY (plugin_id) REFERENCES plugins (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stats (
                plugin_id TEXT PRIMARY KEY,
                downloads INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0,
                reviews_count INTEGER DEFAULT 0,
                last_updated TEXT,
                compatibility_score REAL DEFAULT 0.0,
                performance_score REAL DEFAULT 0.0,
                FOREIGN KEY (plugin_id) REFERENCES plugins (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def submit_plugin(self, plugin_data: Dict[str, Any], 
                     plugin_file: str, author_id: str) -> str:
        """
        Submit a plugin to the marketplace.
        
        Args:
            plugin_data: Plugin metadata
            plugin_file: Path to plugin file
            author_id: ID of the author
            
        Returns:
            Plugin ID
        """
        # Generate plugin ID
        plugin_id = self._generate_plugin_id(plugin_data['name'], plugin_data['version'])
        
        # Validate plugin
        validation_result = self._validate_plugin(plugin_file, plugin_data)
        if not validation_result['valid']:
            raise ValueError(f"Plugin validation failed: {validation_result['errors']}")
        
        # Calculate checksum
        checksum = self._calculate_checksum(plugin_file)
        
        # Get file size
        file_size = os.path.getsize(plugin_file)
        
        # Create plugin record
        plugin = MarketplacePlugin(
            id=plugin_id,
            name=plugin_data['name'],
            version=plugin_data['version'],
            description=plugin_data['description'],
            author=plugin_data['author'],
            category=PluginCategory(plugin_data['category']),
            status=PluginStatus.PENDING,
            tags=plugin_data.get('tags', []),
            dependencies=plugin_data.get('dependencies', []),
            license=plugin_data.get('license', 'MIT'),
            homepage=plugin_data.get('homepage'),
            repository=plugin_data.get('repository'),
            documentation=plugin_data.get('documentation'),
            created_date=datetime.now(),
            updated_date=datetime.now(),
            reviews=[],
            stats=PluginStats(
                downloads=0,
                rating=0.0,
                reviews_count=0,
                last_updated=datetime.now(),
                compatibility_score=0.0,
                performance_score=0.0
            ),
            download_url=f"/plugins/{plugin_id}",
            checksum=checksum,
            file_size=file_size,
            compatibility={}
        )
        
        # Save plugin file
        plugin_path = self.plugins_dir / f"{plugin_id}.zip"
        shutil.copy2(plugin_file, plugin_path)
        
        # Save to database
        self._save_plugin_to_db(plugin)
        
        logger.info(f"Plugin {plugin_id} submitted by {author_id}")
        return plugin_id
    
    def _generate_plugin_id(self, name: str, version: str) -> str:
        """Generate unique plugin ID."""
        # Create hash from name, version, and timestamp
        content = f"{name}_{version}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _validate_plugin(self, plugin_file: str, plugin_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate plugin before submission."""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['name', 'version', 'description', 'author', 'category']
        for field in required_fields:
            if field not in plugin_data or not plugin_data[field]:
                errors.append(f"Missing required field: {field}")
        
        # Check file exists and is valid
        if not os.path.exists(plugin_file):
            errors.append("Plugin file not found")
        elif not plugin_file.endswith('.zip'):
            warnings.append("Plugin file should be a ZIP archive")
        
        # Check plugin structure
        if os.path.exists(plugin_file):
            try:
                # Extract and check plugin structure
                with tempfile.TemporaryDirectory() as temp_dir:
                    shutil.unpack_archive(plugin_file, temp_dir)
                    
                    # Check for required files
                    required_files = ['metadata.json']
                    for file in required_files:
                        if not os.path.exists(os.path.join(temp_dir, file)):
                            errors.append(f"Missing required file: {file}")
                    
                    # Check for Python files
                    python_files = [f for f in os.listdir(temp_dir) if f.endswith('.py')]
                    if not python_files:
                        errors.append("No Python files found in plugin")
                    
            except Exception as e:
                errors.append(f"Error validating plugin structure: {e}")
        
        # Check version format
        version = plugin_data.get('version', '')
        if version and not self._is_valid_version(version):
            errors.append("Invalid version format (use semantic versioning)")
        
        # Check license
        license_type = plugin_data.get('license', '')
        valid_licenses = ['MIT', 'Apache-2.0', 'GPL-3.0', 'BSD-3-Clause', 'ISC']
        if license_type and license_type not in valid_licenses:
            warnings.append(f"Uncommon license: {license_type}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _is_valid_version(self, version: str) -> bool:
        """Check if version follows semantic versioning."""
        import re
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?(\+[a-zA-Z0-9]+)?$'
        return bool(re.match(pattern, version))
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _save_plugin_to_db(self, plugin: MarketplacePlugin):
        """Save plugin to database."""
        if not SQLITE_AVAILABLE:
            # Fallback to file-based storage
            self._save_plugin_to_file(plugin)
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO plugins (
                id, name, version, description, author, category, status,
                tags, dependencies, license, homepage, repository, documentation,
                created_date, updated_date, download_url, checksum, file_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            plugin.id, plugin.name, plugin.version, plugin.description,
            plugin.author, plugin.category.value, plugin.status.value,
            json.dumps(plugin.tags), json.dumps(plugin.dependencies),
            plugin.license, plugin.homepage, plugin.repository,
            plugin.documentation, plugin.created_date.isoformat(),
            plugin.updated_date.isoformat(), plugin.download_url,
            plugin.checksum, plugin.file_size
        ))
        
        # Save stats
        cursor.execute('''
            INSERT OR REPLACE INTO stats (
                plugin_id, downloads, rating, reviews_count, last_updated,
                compatibility_score, performance_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            plugin.id, plugin.stats.downloads, plugin.stats.rating,
            plugin.stats.reviews_count, plugin.stats.last_updated.isoformat(),
            plugin.stats.compatibility_score, plugin.stats.performance_score
        ))
        
        conn.commit()
        conn.close()
    
    def _save_plugin_to_file(self, plugin: MarketplacePlugin):
        """Save plugin to file (fallback when SQLite not available)."""
        plugin_file = self.plugins_dir / f"{plugin.id}.json"
        with open(plugin_file, 'w') as f:
            json.dump(asdict(plugin), f, indent=2, default=str)
    
    def get_plugin(self, plugin_id: str) -> Optional[MarketplacePlugin]:
        """Get plugin by ID."""
        if not SQLITE_AVAILABLE:
            return self._get_plugin_from_file(plugin_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM plugins WHERE id = ?', (plugin_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Convert row to plugin object
        plugin = self._row_to_plugin(row)
        
        # Get reviews
        cursor.execute('SELECT * FROM reviews WHERE plugin_id = ?', (plugin_id,))
        review_rows = cursor.fetchall()
        plugin.reviews = [self._row_to_review(row) for row in review_rows]
        
        # Get stats
        cursor.execute('SELECT * FROM stats WHERE plugin_id = ?', (plugin_id,))
        stats_row = cursor.fetchone()
        if stats_row:
            plugin.stats = self._row_to_stats(stats_row)
        
        conn.close()
        return plugin
    
    def _get_plugin_from_file(self, plugin_id: str) -> Optional[MarketplacePlugin]:
        """Get plugin from file (fallback)."""
        plugin_file = self.plugins_dir / f"{plugin_id}.json"
        if not plugin_file.exists():
            return None
        
        with open(plugin_file, 'r') as f:
            data = json.load(f)
            return MarketplacePlugin(**data)
    
    def _row_to_plugin(self, row) -> MarketplacePlugin:
        """Convert database row to plugin object."""
        return MarketplacePlugin(
            id=row[0],
            name=row[1],
            version=row[2],
            description=row[3],
            author=row[4],
            category=PluginCategory(row[5]),
            status=PluginStatus(row[6]),
            tags=json.loads(row[7]) if row[7] else [],
            dependencies=json.loads(row[8]) if row[8] else [],
            license=row[9],
            homepage=row[10],
            repository=row[11],
            documentation=row[12],
            created_date=datetime.fromisoformat(row[13]),
            updated_date=datetime.fromisoformat(row[14]),
            reviews=[],
            stats=PluginStats(
                downloads=0, rating=0.0, reviews_count=0,
                last_updated=datetime.now(), compatibility_score=0.0,
                performance_score=0.0
            ),
            download_url=row[15],
            checksum=row[16],
            file_size=row[17],
            compatibility={}
        )
    
    def _row_to_review(self, row) -> PluginReview:
        """Convert database row to review object."""
        return PluginReview(
            reviewer_id=row[2],
            reviewer_name=row[3],
            review_date=datetime.fromisoformat(row[4]),
            status=ReviewStatus(row[5]),
            score=row[6],
            comments=row[7],
            technical_review=row[8],
            security_review=row[9],
            performance_review=row[10],
            documentation_review=row[11]
        )
    
    def _row_to_stats(self, row) -> PluginStats:
        """Convert database row to stats object."""
        return PluginStats(
            downloads=row[1],
            rating=row[2],
            reviews_count=row[3],
            last_updated=datetime.fromisoformat(row[4]),
            compatibility_score=row[5],
            performance_score=row[6]
        )
    
    def search_plugins(self, query: str = "", category: Optional[PluginCategory] = None,
                      status: Optional[PluginStatus] = None, 
                      min_rating: float = 0.0) -> List[MarketplacePlugin]:
        """
        Search plugins in the marketplace.
        
        Args:
            query: Search query
            category: Filter by category
            status: Filter by status
            min_rating: Minimum rating filter
            
        Returns:
            List of matching plugins
        """
        if not SQLITE_AVAILABLE:
            return self._search_plugins_from_files(query, category, status, min_rating)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        where_conditions = []
        params = []
        
        if query:
            where_conditions.append("(name LIKE ? OR description LIKE ? OR tags LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
        
        if category:
            where_conditions.append("category = ?")
            params.append(category.value)
        
        if status:
            where_conditions.append("status = ?")
            params.append(status.value)
        
        if min_rating > 0:
            where_conditions.append("id IN (SELECT plugin_id FROM stats WHERE rating >= ?)")
            params.append(min_rating)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        cursor.execute(f'''
            SELECT p.*, s.rating, s.downloads, s.reviews_count
            FROM plugins p
            LEFT JOIN stats s ON p.id = s.plugin_id
            WHERE {where_clause}
            ORDER BY s.rating DESC, s.downloads DESC
        ''', params)
        
        rows = cursor.fetchall()
        plugins = []
        
        for row in rows:
            plugin = self._row_to_plugin(row)
            # Add rating info
            plugin.stats.rating = row[-3] or 0.0
            plugin.stats.downloads = row[-2] or 0
            plugin.stats.reviews_count = row[-1] or 0
            plugins.append(plugin)
        
        conn.close()
        return plugins
    
    def _search_plugins_from_files(self, query: str, category: Optional[PluginCategory],
                                  status: Optional[PluginStatus], 
                                  min_rating: float) -> List[MarketplacePlugin]:
        """Search plugins from files (fallback)."""
        plugins = []
        
        for plugin_file in self.plugins_dir.glob("*.json"):
            try:
                with open(plugin_file, 'r') as f:
                    data = json.load(f)
                    plugin = MarketplacePlugin(**data)
                    
                    # Apply filters
                    if query and query.lower() not in plugin.name.lower() and query.lower() not in plugin.description.lower():
                        continue
                    
                    if category and plugin.category != category:
                        continue
                    
                    if status and plugin.status != status:
                        continue
                    
                    if min_rating > 0 and plugin.stats.rating < min_rating:
                        continue
                    
                    plugins.append(plugin)
                    
            except Exception as e:
                logger.warning(f"Error loading plugin from {plugin_file}: {e}")
        
        return plugins
    
    def review_plugin(self, plugin_id: str, reviewer_id: str, 
                      review_data: Dict[str, Any]) -> bool:
        """
        Submit a review for a plugin.
        
        Args:
            plugin_id: Plugin ID
            reviewer_id: Reviewer ID
            review_data: Review data
            
        Returns:
            Success status
        """
        if not SQLITE_AVAILABLE:
            return self._review_plugin_to_file(plugin_id, reviewer_id, review_data)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert review
        cursor.execute('''
            INSERT INTO reviews (
                plugin_id, reviewer_id, reviewer_name, review_date, status,
                score, comments, technical_review, security_review,
                performance_review, documentation_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            plugin_id, reviewer_id, review_data['reviewer_name'],
            datetime.now().isoformat(), ReviewStatus.PENDING.value,
            review_data['score'], review_data['comments'],
            review_data.get('technical_review', ''),
            review_data.get('security_review', ''),
            review_data.get('performance_review', ''),
            review_data.get('documentation_review', '')
        ))
        
        # Update plugin stats
        cursor.execute('''
            UPDATE stats SET reviews_count = reviews_count + 1
            WHERE plugin_id = ?
        ''', (plugin_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Review submitted for plugin {plugin_id} by {reviewer_id}")
        return True
    
    def _review_plugin_to_file(self, plugin_id: str, reviewer_id: str, 
                              review_data: Dict[str, Any]) -> bool:
        """Save review to file (fallback)."""
        # This is a simplified implementation
        # In practice, you'd want to store reviews in a separate file or database
        logger.info(f"Review submitted for plugin {plugin_id} by {reviewer_id}")
        return True
    
    def download_plugin(self, plugin_id: str, user_id: str) -> Optional[str]:
        """
        Download a plugin.
        
        Args:
            plugin_id: Plugin ID
            user_id: User ID
            
        Returns:
            Path to downloaded plugin file
        """
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            return None
        
        # Update download stats
        self._update_download_stats(plugin_id)
        
        # Return plugin file path
        plugin_file = self.plugins_dir / f"{plugin_id}.zip"
        if plugin_file.exists():
            logger.info(f"Plugin {plugin_id} downloaded by {user_id}")
            return str(plugin_file)
        
        return None
    
    def _update_download_stats(self, plugin_id: str):
        """Update download statistics."""
        if not SQLITE_AVAILABLE:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE stats SET downloads = downloads + 1, last_updated = ?
            WHERE plugin_id = ?
        ''', (datetime.now().isoformat(), plugin_id))
        
        conn.commit()
        conn.close()
    
    def _create_web_app(self) -> Flask:
        """Create Flask web application."""
        app = Flask(__name__)
        CORS(app)
        app.secret_key = 'your-secret-key-here'  # Change this in production
        
        @app.route('/')
        def index():
            return render_template('marketplace.html')
        
        @app.route('/api/plugins')
        def get_plugins():
            query = request.args.get('q', '')
            category = request.args.get('category')
            status = request.args.get('status')
            min_rating = float(request.args.get('min_rating', 0))
            
            category_enum = PluginCategory(category) if category else None
            status_enum = PluginStatus(status) if status else None
            
            plugins = self.search_plugins(query, category_enum, status_enum, min_rating)
            
            return jsonify({
                'plugins': [asdict(plugin) for plugin in plugins]
            })
        
        @app.route('/api/plugins/<plugin_id>')
        def get_plugin(plugin_id):
            plugin = self.get_plugin(plugin_id)
            if not plugin:
                return jsonify({'error': 'Plugin not found'}), 404
            
            return jsonify(asdict(plugin))
        
        @app.route('/api/plugins/<plugin_id>/download')
        def download_plugin(plugin_id):
            user_id = session.get('user_id', 'anonymous')
            plugin_path = self.download_plugin(plugin_id, user_id)
            
            if not plugin_path:
                return jsonify({'error': 'Plugin not found'}), 404
            
            return send_file(plugin_path, as_attachment=True)
        
        @app.route('/api/plugins/<plugin_id>/review', methods=['POST'])
        def submit_review(plugin_id):
            review_data = request.json
            reviewer_id = session.get('user_id', 'anonymous')
            
            success = self.review_plugin(plugin_id, reviewer_id, review_data)
            
            if success:
                return jsonify({'success': True})
            else:
                return jsonify({'error': 'Failed to submit review'}), 500
        
        @app.route('/api/plugins/submit', methods=['POST'])
        def submit_plugin():
            # Handle plugin submission
            plugin_data = request.form.to_dict()
            author_id = session.get('user_id', 'anonymous')
            
            # Handle file upload
            if 'plugin_file' not in request.files:
                return jsonify({'error': 'No plugin file provided'}), 400
            
            file = request.files['plugin_file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save uploaded file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            file.save(temp_file.name)
            
            try:
                plugin_id = self.submit_plugin(plugin_data, temp_file.name, author_id)
                return jsonify({'success': True, 'plugin_id': plugin_id})
            except Exception as e:
                return jsonify({'error': str(e)}), 400
            finally:
                os.unlink(temp_file.name)
        
        return app
    
    def start_web_server(self, host: str = 'localhost', port: int = 5000, 
                        debug: bool = False) -> None:
        """Start the web server."""
        if not self.app:
            raise RuntimeError("Web interface not available (Flask not installed)")
        
        print(f"Starting plugin marketplace at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


class PluginQualityControl:
    """
    Quality control system for plugin marketplace.
    """
    
    def __init__(self, marketplace: PluginMarketplace):
        """
        Initialize quality control system.
        
        Args:
            marketplace: Plugin marketplace instance
        """
        self.marketplace = marketplace
        self.reviewers = {}
        self.quality_metrics = {}
    
    def assign_reviewer(self, plugin_id: str, reviewer_id: str) -> bool:
        """Assign a reviewer to a plugin."""
        self.reviewers[plugin_id] = reviewer_id
        logger.info(f"Reviewer {reviewer_id} assigned to plugin {plugin_id}")
        return True
    
    def perform_automated_review(self, plugin_id: str) -> Dict[str, Any]:
        """
        Perform automated quality review of a plugin.
        
        Args:
            plugin_id: Plugin ID to review
            
        Returns:
            Automated review results
        """
        plugin = self.marketplace.get_plugin(plugin_id)
        if not plugin:
            return {'error': 'Plugin not found'}
        
        review_results = {
            'plugin_id': plugin_id,
            'automated_score': 0.0,
            'issues': [],
            'recommendations': [],
            'security_issues': [],
            'performance_issues': [],
            'documentation_issues': []
        }
        
        # Check plugin structure
        structure_score = self._check_plugin_structure(plugin)
        review_results['structure_score'] = structure_score
        
        # Check code quality
        code_score = self._check_code_quality(plugin)
        review_results['code_score'] = code_score
        
        # Check documentation
        doc_score = self._check_documentation(plugin)
        review_results['documentation_score'] = doc_score
        
        # Check security
        security_score = self._check_security(plugin)
        review_results['security_score'] = security_score
        
        # Calculate overall score
        overall_score = (structure_score + code_score + doc_score + security_score) / 4
        review_results['automated_score'] = overall_score
        
        # Generate recommendations
        if overall_score < 0.7:
            review_results['recommendations'].append("Plugin needs improvement before approval")
        
        if structure_score < 0.8:
            review_results['issues'].append("Plugin structure needs improvement")
        
        if code_score < 0.7:
            review_results['issues'].append("Code quality needs improvement")
        
        if doc_score < 0.6:
            review_results['documentation_issues'].append("Documentation is insufficient")
        
        if security_score < 0.8:
            review_results['security_issues'].append("Security issues detected")
        
        return review_results
    
    def _check_plugin_structure(self, plugin: MarketplacePlugin) -> float:
        """Check plugin structure and return score."""
        # This is a simplified implementation
        # Full implementation would check actual plugin files
        
        score = 1.0
        
        # Check required metadata
        if not plugin.description:
            score -= 0.2
        
        if not plugin.documentation:
            score -= 0.1
        
        if not plugin.tags:
            score -= 0.1
        
        return max(0.0, score)
    
    def _check_code_quality(self, plugin: MarketplacePlugin) -> float:
        """Check code quality and return score."""
        # This is a simplified implementation
        # Full implementation would analyze actual code
        
        score = 1.0
        
        # Check for common issues
        if plugin.name.lower() in ['test', 'example', 'demo']:
            score -= 0.3
        
        if len(plugin.description) < 10:
            score -= 0.2
        
        return max(0.0, score)
    
    def _check_documentation(self, plugin: MarketplacePlugin) -> float:
        """Check documentation quality and return score."""
        score = 1.0
        
        if not plugin.documentation:
            score -= 0.5
        
        if not plugin.homepage:
            score -= 0.2
        
        if not plugin.repository:
            score -= 0.2
        
        return max(0.0, score)
    
    def _check_security(self, plugin: MarketplacePlugin) -> float:
        """Check security and return score."""
        score = 1.0
        
        # Check license
        if plugin.license not in ['MIT', 'Apache-2.0', 'BSD-3-Clause']:
            score -= 0.1
        
        # Check dependencies
        if len(plugin.dependencies) > 10:
            score -= 0.2
        
        return max(0.0, score)


# Main functions
def create_marketplace(db_path: str = "marketplace.db", 
                      plugins_dir: str = "marketplace_plugins") -> PluginMarketplace:
    """Create a plugin marketplace instance."""
    return PluginMarketplace(db_path, plugins_dir)


def run_marketplace_web(db_path: str = "marketplace.db", 
                       plugins_dir: str = "marketplace_plugins",
                       host: str = 'localhost', port: int = 5000):
    """Run the marketplace web interface."""
    marketplace = create_marketplace(db_path, plugins_dir)
    
    if not marketplace.app:
        print("Web interface not available (Flask not installed)")
        print("Please install Flask: pip install flask flask-cors")
        return
    
    print("Starting plugin marketplace...")
    marketplace.start_web_server(host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Coratrix Plugin Marketplace")
    parser.add_argument("--mode", choices=["web"], default="web", 
                      help="Marketplace mode")
    parser.add_argument("--db-path", default="marketplace.db", 
                      help="Database path")
    parser.add_argument("--plugins-dir", default="marketplace_plugins", 
                      help="Plugins directory")
    parser.add_argument("--host", default="localhost", 
                      help="Host for web interface")
    parser.add_argument("--port", type=int, default=5000, 
                      help="Port for web interface")
    
    args = parser.parse_args()
    
    if args.mode == "web":
        run_marketplace_web(args.db_path, args.plugins_dir, args.host, args.port)
