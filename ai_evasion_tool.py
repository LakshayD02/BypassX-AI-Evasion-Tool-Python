#!/usr/bin/env python3
"""
Advanced AI Evasion Tool - Professional Red Teaming Framework
Author: Security Research Team
Version: 2.3 (Fixed visualization typo)
Description: Advanced tool for testing AI-based security systems with evasion techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats
import random
import string
import json
import yaml
import csv
import time
import logging
import argparse
import socket
import struct
import ipaddress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import base64
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_evasion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AIEvasionTool")

class AdvancedEvasionEngine:
    """Advanced evasion techniques with multiple attack vectors"""
    
    def __init__(self):
        self.techniques = {
            'obfuscation': self._apply_obfuscation,
            'polymorphic': self._apply_polymorphic,
            'metamorphic': self._apply_metamorphic,
            'adversarial': self._apply_adversarial,
            'timing': self._apply_timing_attack,
            'traffic_morphing': self._apply_traffic_morphing,
            'domain_generation': self._generate_dga_domains,
            'encryption': self._apply_encryption
        }
    
    def _apply_obfuscation(self, payload: str) -> str:
        """Apply multiple obfuscation techniques"""
        techniques = [
            self._base64_obfuscate,
            self._hex_encode,
            self._rot13_encode,
            self._string_replacements
        ]
        obfuscated = payload
        for technique in random.sample(techniques, random.randint(2, 4)):
            obfuscated = technique(obfuscated)
        return obfuscated
    
    def _base64_obfuscate(self, payload: str) -> str:
        return base64.b64encode(payload.encode()).decode()
    
    def _hex_encode(self, payload: str) -> str:
        return payload.encode().hex()
    
    def _rot13_encode(self, payload: str) -> str:
        result = []
        for char in payload:
            if 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + 13) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + 13) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)
    
    def _string_replacements(self, payload: str) -> str:
        replacements = {
            'http': 'hxxp',
            '.': '[.]',
            '@': '[at]',
            'https': 'hxxps',
            'www': 'vvv'
        }
        for old, new in replacements.items():
            payload = payload.replace(old, new)
        return payload
    
    def _apply_polymorphic(self, payload: str) -> str:
        """Generate polymorphic code variations"""
        junk_code = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(10, 30)))
        polymorphic_var = random.randint(1000, 9999)
        return f"{junk_code}{payload}{polymorphic_var}"
    
    def _apply_metamorphic(self, payload: str) -> str:
        """Apply metamorphic techniques that change code structure"""
        # Simulate code restructuring
        lines = payload.split('\n')
        random.shuffle(lines)
        return '\n'.join(lines)
    
    def _apply_adversarial(self, feature_vector: np.ndarray, model: Any, epsilon: float = 0.1) -> np.ndarray:
        """Apply FGSM adversarial attack"""
        if not hasattr(model, 'predict_proba'):
            return feature_vector
            
        tensor = torch.tensor(feature_vector, dtype=torch.float32, requires_grad=True)
        output = torch.tensor(model.predict_proba(feature_vector))
        
        loss = nn.CrossEntropyLoss()(output, torch.argmax(output, dim=1))
        loss.backward()
        
        perturbation = epsilon * tensor.grad.data.sign()
        return (tensor + perturbation).detach().numpy()
    
    def _apply_timing_attack(self, payload: str) -> Dict:
        """Add timing characteristics to evade behavioral analysis"""
        # Return a dictionary with timing information and the original payload
        return {
            'payload': payload,
            'timing': {
                'delay': random.uniform(0.5, 5.0),
                'jitter': random.uniform(0.1, 1.0),
                'burst_mode': random.choice([True, False])
            }
        }
    
    def _apply_traffic_morphing(self, payload: str) -> Dict:
        """Morph network traffic characteristics"""
        # Return a dictionary with traffic morphing information
        return {
            'payload': payload,
            'traffic': {
                'ttl': random.randint(50, 128),
                'window_size': random.randint(1024, 65535),
                'tos': random.randint(0, 255)
            }
        }
    
    def _generate_dga_domains(self, seed: str, count: int = 5) -> List[str]:
        """Domain Generation Algorithm for evasion"""
        domains = []
        random.seed(seed)
        tlds = ['.com', '.net', '.org', '.info', '.biz']
        
        for _ in range(count):
            length = random.randint(8, 20)
            domain = ''.join(random.choices(string.ascii_lowercase, k=length))
            domains.append(domain + random.choice(tlds))
        
        return domains
    
    def _apply_encryption(self, payload: str) -> str:
        """Simple XOR encryption for demonstration"""
        key = random.randint(1, 255)
        encrypted = ''.join([chr(ord(c) ^ key) for c in payload])
        return f"{key:02x}{encrypted}"

class AttackVectorManager:
    """Manage different types of attack vectors"""
    
    def __init__(self):
        self.vectors = {
            'phishing': self._create_phishing_vector,
            'malware': self._create_malware_vector,
            'dos': self._create_dos_vector,
            'recon': self._create_recon_vector,
            'lateral_movement': self._create_lateral_movement_vector,
            'data_exfiltration': self._create_data_exfiltration_vector
        }
    
    def _create_phishing_vector(self, base_payload: str) -> Dict:
        """Create phishing attack vector"""
        return {
            'type': 'phishing',
            'payload': base_payload,
            'headers': {
                'User-Agent': random.choice([
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15'
                ]),
                'Content-Type': 'text/html'
            },
            'target': 'web_application'
        }
    
    def _create_malware_vector(self, base_payload: str) -> Dict:
        """Create malware payload vector"""
        return {
            'type': 'malware',
            'payload': base_payload,
            'format': random.choice(['exe', 'dll', 'script', 'document']),
            'execution_method': random.choice(['process_injection', 'scheduled_task', 'registry'])
        }
    
    def _create_dos_vector(self, base_payload: str) -> Dict:
        """Create Denial of Service vector"""
        return {
            'type': 'dos',
            'payload': base_payload,
            'rate': random.randint(100, 1000),
            'duration': random.randint(60, 300),
            'protocol': random.choice(['TCP', 'UDP', 'ICMP'])
        }
    
    def _create_recon_vector(self, base_payload: str) -> Dict:
        """Create reconnaissance vector"""
        return {
            'type': 'recon',
            'payload': base_payload,
            'scan_type': random.choice(['port_scan', 'service_detection', 'os_fingerprinting']),
            'stealth_level': random.choice(['low', 'medium', 'high'])
        }
    
    def _create_lateral_movement_vector(self, base_payload: str) -> Dict:
        """Create lateral movement vector"""
        return {
            'type': 'lateral_movement',
            'payload': base_payload,
            'technique': random.choice(['psexec', 'wmi', 'smb', 'rdp']),
            'target_type': random.choice(['workstation', 'server', 'domain_controller'])
        }
    
    def _create_data_exfiltration_vector(self, base_payload: str) -> Dict:
        """Create data exfiltration vector"""
        return {
            'type': 'data_exfiltration',
            'payload': base_payload,
            'method': random.choice(['dns_tunneling', 'http_post', 'icmp', 'https']),
            'data_size': random.randint(1024, 1048576)
        }

class AIDetectionSimulator:
    """Simulate various AI-based detection systems"""
    
    def __init__(self):
        self.models = {}
        self._train_models()
    
    def _train_models(self):
        """Train multiple detection models"""
        # Synthetic training data
        X_normal = np.random.normal(0, 1, (1000, 10))
        X_anomalous = np.random.normal(3, 2, (200, 10))
        
        X = np.vstack([X_normal, X_anomalous])
        y = np.hstack([np.zeros(1000), np.ones(200)])
        
        # Train multiple models
        self.models['random_forest'] = RandomForestClassifier(n_estimators=100)
        self.models['isolation_forest'] = IsolationForest(contamination=0.2)
        self.models['one_class_svm'] = OneClassSVM(nu=0.1)
        
        self.models['random_forest'].fit(X, y)
        self.models['isolation_forest'].fit(X)
        self.models['one_class_svm'].fit(X_normal)  # Only trained on normal data
    
    def detect(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Run detection across all models"""
        results = {}
        
        # Random Forest
        if hasattr(self.models['random_forest'], 'predict_proba'):
            rf_score = self.models['random_forest'].predict_proba(feature_vector)[0, 1]
            results['rf_score'] = float(rf_score)  # Convert to native Python float
        
        # Isolation Forest
        if_score = -self.models['isolation_forest'].score_samples(feature_vector)[0]
        results['if_score'] = float(if_score)  # Convert to native Python float
        
        # One-Class SVM
        svm_score = -self.models['one_class_svm'].score_samples(feature_vector)[0]
        results['svm_score'] = float(svm_score)  # Convert to native Python float
        
        # Combined score
        results['combined_score'] = float(np.mean(list(results.values())))
        
        return results

class AdvancedAIEvasionTool:
    """Advanced AI Evasion Tool with professional features"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.evasion_engine = AdvancedEvasionEngine()
        self.attack_manager = AttackVectorManager()
        self.detector = AIDetectionSimulator()
        self.logs = []
        self.results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.output_dir = Path(f"results_{self.session_id}")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Advanced AI Evasion Tool initialized. Session ID: {self.session_id}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file"""
        default_config = {
            'evasion_techniques': ['obfuscation', 'polymorphic', 'adversarial', 'timing'],
            'attack_vectors': ['phishing', 'malware', 'dos', 'recon'],
            'max_iterations': 10,
            'detection_threshold': 0.5,
            'report_format': ['console', 'json', 'csv', 'html']
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        return {**default_config, **yaml.safe_load(f)}
                    elif config_path.endswith('.json'):
                        return {**default_config, **json.load(f)}
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using default configuration.")
        
        return default_config
    
    def generate_evasive_payload(self, original_payload: str, techniques: List[str]) -> Dict:
        """Generate evasive payload using multiple techniques"""
        evasive_payload = original_payload
        applied_techniques = []
        
        for technique in techniques:
            if technique in self.evasion_engine.techniques:
                if technique in ['timing', 'traffic_morphing']:
                    # These techniques return a dictionary, not a string
                    result = self.evasion_engine.techniques[technique](evasive_payload)
                    evasive_payload = result['payload']
                    applied_techniques.append(technique)
                elif technique == 'adversarial':
                    # Adversarial requires special handling with feature vectors
                    applied_techniques.append(technique)
                else:
                    # Regular string-based techniques
                    evasive_payload = self.evasion_engine.techniques[technique](evasive_payload)
                    applied_techniques.append(technique)
        
        return {
            'original': original_payload,
            'evasive': evasive_payload,
            'techniques_used': applied_techniques,
            'timestamp': datetime.now().isoformat()
        }
    
    def simulate_attack_campaign(self, attack_type: str, base_payload: str, iterations: int = 5):
        """Simulate a complete attack campaign with multiple iterations"""
        logger.info(f"Starting attack campaign: {attack_type}")
        
        campaign_results = []
        
        for iteration in range(iterations):
            # Select random evasion techniques
            techniques = random.sample(
                self.config['evasion_techniques'],
                random.randint(2, len(self.config['evasion_techniques']))
            )
            
            # Generate evasive payload
            payload_data = self.generate_evasive_payload(base_payload, techniques)
            
            # Create attack vector
            attack_vector = self.attack_manager.vectors[attack_type](payload_data['evasive'])
            
            # Simulate detection
            feature_vector = np.random.rand(1, 10)  # Simulated feature extraction
            detection_results = self.detector.detect(feature_vector)
            
            # Apply adversarial attack if detection is high
            if detection_results['combined_score'] > self.config['detection_threshold'] and 'adversarial' in techniques:
                adversarial_vector = self.evasion_engine._apply_adversarial(
                    feature_vector, self.detector.models['random_forest']
                )
                detection_results = self.detector.detect(adversarial_vector)
            
            # Record results - ensure all values are JSON serializable
            result = {
                'campaign_id': f"{attack_type}_{self.session_id}",
                'iteration': iteration,
                'attack_type': attack_type,
                'detection_scores': detection_results,
                'evasion_success': bool(detection_results['combined_score'] < self.config['detection_threshold']),
                'techniques_used': techniques,
                'timestamp': datetime.now().isoformat()
            }
            
            campaign_results.append(result)
            self.results.append(result)
            
            logger.info(
                f"Iteration {iteration}: Detection Score = {detection_results['combined_score']:.3f}, "
                f"Success = {result['evasion_success']}"
            )
            
            # Add delay between iterations for realism
            time.sleep(random.uniform(0.1, 1.0))
        
        return campaign_results
    
    def run_comprehensive_test(self):
        """Run comprehensive testing across all attack vectors"""
        test_payloads = {
            'phishing': "http://malicious-domain.com/login.php?user=admin&pass=123456",
            'malware': "powershell -ExecutionPolicy Bypass -WindowStyle Hidden -EncodedCommand",
            'dos': "GET / HTTP/1.1\r\nHost: target.com\r\n\r\n" * 100,
            'recon': "nmap -sS -T4 -A -v target.com",
            'lateral_movement': "psexec \\\\target01 cmd.exe",
            'data_exfiltration': "DNS query for large-data-base64-encoded.example.com"
        }
        
        comprehensive_results = {}
        
        for attack_type in self.config['attack_vectors']:
            if attack_type in test_payloads:
                results = self.simulate_attack_campaign(
                    attack_type, test_payloads[attack_type], self.config['max_iterations']
                )
                comprehensive_results[attack_type] = results
        
        return comprehensive_results
    
    def generate_advanced_report(self):
        """Generate comprehensive reports in multiple formats"""
        if not self.results:
            logger.warning("No results available for reporting")
            return
        
        df = pd.DataFrame(self.results)
        
        # Generate reports in all requested formats
        if 'console' in self.config['report_format']:
            self._generate_console_report(df)
        
        if 'json' in self.config['report_format']:
            self._generate_json_report(df)
        
        if 'csv' in self.config['report_format']:
            self._generate_csv_report(df)
        
        if 'html' in self.config['report_format']:
            self._generate_html_report(df)
        
        # Generate visualizations
        self._generate_visualizations(df)
        
        logger.info(f"Reports generated in directory: {self.output_dir}")
    
    def _generate_console_report(self, df: pd.DataFrame):
        """Generate console-based report"""
        print("=" * 80)
        print("ADVANCED AI EVASION TOOL - COMPREHENSIVE REPORT")
        print("=" * 80)
        print(f"Session ID: {self.session_id}")
        print(f"Generated at: {datetime.now().isoformat()}")
        print(f"Total test iterations: {len(df)}")
        print(f"Overall evasion success rate: {df['evasion_success'].mean():.2%}")
        print("\n")
        
        # Technique effectiveness
        all_techniques = [tech for sublist in df['techniques_used'] for tech in sublist]
        tech_effectiveness = {}
        
        for tech in set(all_techniques):
            tech_df = df[df['techniques_used'].apply(lambda x: tech in x)]
            success_rate = tech_df['evasion_success'].mean()
            tech_effectiveness[tech] = success_rate
        
        print("Technique Effectiveness:")
        for tech, success_rate in sorted(tech_effectiveness.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tech:.<20} {success_rate:.2%}")
        
        print("\nAttack Type Analysis:")
        attack_success = df.groupby('attack_type')['evasion_success'].mean()
        for attack, success_rate in attack_success.items():
            print(f"  {attack:.<20} {success_rate:.2%}")
    
    def _generate_json_report(self, df: pd.DataFrame):
        """Generate JSON report with proper serialization"""
        # Convert results to JSON-serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = {}
            for key, value in result.items():
                if key == 'detection_scores':
                    # Ensure all scores are native Python types
                    serializable_result[key] = {k: float(v) for k, v in value.items()}
                elif key == 'evasion_success':
                    # Convert boolean to string for better JSON compatibility
                    serializable_result[key] = bool(value)
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        report = {
            'metadata': {
                'session_id': self.session_id,
                'generation_time': datetime.now().isoformat(),
                'total_iterations': len(df)
            },
            'summary': {
                'overall_success_rate': float(df['evasion_success'].mean()),
                'attack_type_success': {k: float(v) for k, v in df.groupby('attack_type')['evasion_success'].mean().to_dict().items()},
                'technique_effectiveness': {
                    tech: float(df[df['techniques_used'].apply(lambda x: tech in x)]['evasion_success'].mean())
                    for tech in set([tech for sublist in df['techniques_used'] for tech in sublist])
                }
            },
            'detailed_results': serializable_results
        }
        
        with open(self.output_dir / 'detailed_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _generate_csv_report(self, df: pd.DataFrame):
        """Generate CSV report"""
        # Flatten detection scores
        flattened_data = []
        for result in self.results:
            row = {
                'campaign_id': result['campaign_id'],
                'iteration': result['iteration'],
                'attack_type': result['attack_type'],
                'evasion_success': result['evasion_success'],
                'techniques_used': '|'.join(result['techniques_used']),
                'timestamp': result['timestamp']
            }
            row.update({f'score_{k}': v for k, v in result['detection_scores'].items()})
            flattened_data.append(row)
        
        csv_df = pd.DataFrame(flattened_data)
        csv_df.to_csv(self.output_dir / 'detailed_results.csv', index=False)
    
    def _generate_html_report(self, df: pd.DataFrame):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Evasion Tool Report - {self.session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Evasion Tool Report</h1>
                <p>Session ID: {self.session_id}</p>
                <p>Generated: {datetime.now().isoformat()}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {len(df)}</p>
                <p>Success Rate: {df['evasion_success'].mean():.2%}</p>
            </div>
            
            <h2>Results by Attack Type</h2>
            <table class="table">
                <tr><th>Attack Type</th><th>Success Rate</th><th>Test Count</th></tr>
        """
        
        for attack_type, group in df.groupby('attack_type'):
            success_rate = group['evasion_success'].mean()
            html_content += f"""
                <tr>
                    <td>{attack_type}</td>
                    <td class="{ 'success' if success_rate > 0.7 else 'failure' }">{success_rate:.2%}</td>
                    <td>{len(group)}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Technique Effectiveness</h2>
            <table class="table">
                <tr><th>Technique</th><th>Success Rate</th><th>Usage Count</th></tr>
        """
        
        # Calculate technique effectiveness
        all_techniques = [tech for sublist in df['techniques_used'] for tech in sublist]
        tech_counts = pd.Series(all_techniques).value_counts()
        
        for tech, count in tech_counts.items():
            success_rate = df[df['techniques_used'].apply(lambda x: tech in x)]['evasion_success'].mean()
            html_content += f"""
                <tr>
                    <td>{tech}</td>
                    <td class="{ 'success' if success_rate > 0.7 else 'failure' }">{success_rate:.2%}</td>
                    <td>{count}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'report.html', 'w') as f:
            f.write(html_content)
    
    def _generate_visualizations(self, df: pd.DataFrame):
        """Generate comprehensive visualizations of the results"""
        plt.style.use('seaborn-v0_8')
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'AI Evasion Tool Results - Session {self.session_id}', fontsize=16, fontweight='bold')
        
        # 1. Success rate by attack type
        attack_success = df.groupby('attack_type')['evasion_success'].mean()
        axes[0, 0].bar(attack_success.index, attack_success.values, color=['green' if x > 0.5 else 'red' for x in attack_success.values])
        axes[0, 0].set_title('Success Rate by Attack Type')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Technique effectiveness
        all_techniques = [tech for sublist in df['techniques_used'] for tech in sublist]
        tech_effectiveness = {}
        for tech in set(all_techniques):
            tech_effectiveness[tech] = df[df['techniques_used'].apply(lambda x: tech in x)]['evasion_success'].mean()
        
        tech_names = list(tech_effectiveness.keys())
        tech_rates = list(tech_effectiveness.values())
        axes[0, 1].barh(tech_names, tech_rates, color=['green' if x > 0.5 else 'red' for x in tech_rates])
        axes[0, 1].set_title('Technique Effectiveness')
        axes[0, 1].set_xlabel('Success Rate')
        
        # 3. Detection score distribution
        combined_scores = [r['detection_scores']['combined_score'] for r in self.results]
        axes[0, 2].hist(combined_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 2].axvline(self.config['detection_threshold'], color='red', linestyle='--', label='Detection Threshold')
        axes[0, 2].set_title('Detection Score Distribution')
        axes[0, 2].set_xlabel('Detection Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        
        # 4. Success rate over time (iterations)
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp_dt')
        success_cumulative = df_sorted['evasion_success'].expanding().mean()
        axes[1, 0].plot(range(len(success_cumulative)), success_cumulative, marker='o', linestyle='-')
        axes[1, 0].set_title('Cumulative Success Rate Over Time')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Cumulative Success Rate')
        axes[1, 0].grid(True)
        
        # 5. Model performance comparison
        model_scores = {
            'Random Forest': [r['detection_scores']['rf_score'] for r in self.results],
            'Isolation Forest': [r['detection_scores']['if_score'] for r in self.results],
            'One-Class SVM': [r['detection_scores']['svm_score'] for r in self.results]
        }
        model_means = [np.mean(scores) for scores in model_scores.values()]
        axes[1, 1].bar(model_scores.keys(), model_means, color=['blue', 'orange', 'green'])
        axes[1, 1].set_title('Average Detection Score by Model')
        axes[1, 1].set_ylabel('Average Detection Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Technique combination heatmap
        tech_combinations = {}
        for techniques in df['techniques_used']:
            if len(techniques) >= 2:
                key = '+'.join(sorted(techniques))
                tech_combinations[key] = tech_combinations.get(key, 0) + 1
        
        if tech_combinations:
            comb_names = list(tech_combinations.keys())[:10]  # Top 10 combinations
            comb_counts = list(tech_combinations.values())[:10]
            axes[1, 2].barh(comb_names, comb_counts, color='purple')
            axes[1, 2].set_title('Top Technique Combinations')
            axes[1, 2].set_xlabel('Usage Count')
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data for technique combinations', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Technique Combinations')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualization_report.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the Advanced AI Evasion Tool"""
    parser = argparse.ArgumentParser(description='Advanced AI Evasion Tool - Red Teaming Framework')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--attack', '-a', type=str, choices=[
        'phishing', 'malware', 'dos', 'recon', 'lateral_movement', 'data_exfiltration', 'all'
    ], default='all', help='Specific attack type to test')
    parser.add_argument('--iterations', '-i', type=int, default=10, help='Number of iterations per attack')
    parser.add_argument('--output', '-o', type=str, help='Custom output directory')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (less console output)')
    
    args = parser.parse_args()
    
    # Set logging level based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize the tool
    tool = AdvancedAIEvasionTool(args.config)
    
    # Override config if specific arguments provided
    if args.attack != 'all':
        tool.config['attack_vectors'] = [args.attack]
    
    if args.iterations:
        tool.config['max_iterations'] = args.iterations
    
    if args.output:
        tool.output_dir = Path(args.output)
        tool.output_dir.mkdir(exist_ok=True)
    
    # Run the comprehensive test
    logger.info("Starting comprehensive AI evasion testing")
    results = tool.run_comprehensive_test()
    
    # Generate reports
    logger.info("Generating comprehensive reports")
    tool.generate_advanced_report()
    
    logger.info(f"AI evasion testing completed. Results saved to: {tool.output_dir}")

if __name__ == "__main__":
    main()