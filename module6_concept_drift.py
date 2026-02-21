"""
Module 6: Concept Drift Detection
Implements ADWIN and Page-Hinkley tests for detecting sudden, gradual,
and recurring drift. Includes drift response mechanisms.
"""

import pandas as pd
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class ADWIN:
    """Adaptive Windowing algorithm for drift detection."""
    
    def __init__(self, delta=0.002, min_window_length=10):
        """Initialize ADWIN detector."""
        self.delta = delta
        self.min_window_length = min_window_length
        self.window = deque()
        self.drift_detected = False
        self.drift_points = []
        
    def add_element(self, value):
        """Add new element and check for drift."""
        self.window.append(value)
        self.drift_detected = False
        
        if len(self.window) < self.min_window_length * 2:
            return False
        
        # Try to find cut point
        n = len(self.window)
        for cut in range(self.min_window_length, n - self.min_window_length + 1):
            w0 = list(self.window)[:cut]
            w1 = list(self.window)[cut:]
            
            if len(w0) == 0 or len(w1) == 0:
                continue
            
            mean0 = np.mean(w0)
            mean1 = np.mean(w1)
            var0 = np.var(w0)
            var1 = np.var(w1)
            
            # Calculate threshold
            m = 1 / (1/len(w0) + 1/len(w1))
            dd = np.log(2 * np.log(n) / self.delta)
            epsilon = np.sqrt(2 * m * var0 * dd) + (2/3 * dd * m)
            
            if abs(mean0 - mean1) > epsilon:
                # Drift detected
                self.drift_detected = True
                self.drift_points.append(len(self.window))
                # Remove old window
                for _ in range(cut):
                    self.window.popleft()
                return True
        
        return False
    
    def reset(self):
        """Reset the detector."""
        self.window.clear()
        self.drift_detected = False


class PageHinkley:
    """Page-Hinkley test for drift detection."""
    
    def __init__(self, delta=0.005, alpha=0.99, threshold=50):
        """Initialize Page-Hinkley detector."""
        self.delta = delta
        self.alpha = alpha
        self.threshold = threshold
        self.mean = 0.0
        self.sum = 0.0
        self.min_sum = float('inf')
        self.drift_detected = False
        self.drift_points = []
        self.count = 0
        
    def add_element(self, value):
        """Add new element and check for drift."""
        self.count += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value
        self.sum += (value - self.mean - self.delta)
        self.min_sum = min(self.min_sum, self.sum)
        
        self.drift_detected = False
        
        if self.sum - self.min_sum > self.threshold:
            self.drift_detected = True
            self.drift_points.append(self.count)
            # Reset
            self.sum = 0.0
            self.min_sum = float('inf')
            self.mean = value
            return True
        
        return False
    
    def reset(self):
        """Reset the detector."""
        self.mean = 0.0
        self.sum = 0.0
        self.min_sum = float('inf')
        self.drift_detected = False
        self.count = 0


class ConceptDriftDetector:
    """Main concept drift detection system."""
    
    def __init__(self, adwin_delta=0.002, ph_threshold=50):
        """Initialize drift detection system."""
        self.adwin = ADWIN(delta=adwin_delta)
        self.page_hinkley = PageHinkley(threshold=ph_threshold)
        self.drift_history = []
        self.drift_types = []
        
    def detect_drift(self, value):
        """Detect drift using both methods."""
        adwin_drift = self.adwin.add_element(value)
        ph_drift = self.page_hinkley.add_element(value)
        
        drift_detected = adwin_drift or ph_drift
        
        if drift_detected:
            # Classify drift type
            drift_type = self._classify_drift_type()
            self.drift_history.append({
                'index': len(self.drift_history),
                'value': value,
                'adwin_detected': adwin_drift,
                'ph_detected': ph_drift,
                'drift_type': drift_type
            })
            self.drift_types.append(drift_type)
        
        return drift_detected, adwin_drift, ph_drift
    
    def _classify_drift_type(self):
        """Classify drift type based on detection pattern."""
        if len(self.drift_history) < 2:
            return 'sudden'
        
        # Check if drift is recurring
        recent_drifts = [d['index'] for d in self.drift_history[-5:]]
        if len(recent_drifts) >= 3:
            intervals = [recent_drifts[i+1] - recent_drifts[i] 
                        for i in range(len(recent_drifts)-1)]
            if np.std(intervals) < np.mean(intervals) * 0.3:
                return 'recurring'
        
        # Check if gradual (multiple detections close together)
        if len(self.drift_history) >= 2:
            last_two = self.drift_history[-2:]
            if abs(last_two[1]['index'] - last_two[0]['index']) < 10:
                return 'gradual'
        
        return 'sudden'
    
    def get_drift_summary(self):
        """Get summary of detected drifts."""
        if not self.drift_history:
            return None
        
        summary = {
            'total_drifts': len(self.drift_history),
            'drift_types': pd.Series(self.drift_types).value_counts().to_dict(),
            'adwin_detections': sum(1 for d in self.drift_history if d['adwin_detected']),
            'ph_detections': sum(1 for d in self.drift_history if d['ph_detected'])
        }
        
        return summary


class DriftResponseSystem:
    """System for responding to detected drift."""
    
    def __init__(self, model=None):
        """Initialize drift response system."""
        self.model = model
        self.retraining_threshold = 0.1  # Retrain if performance drops by 10%
        self.alert_threshold = 0.2  # Alert if performance drops by 20%
        
    def respond_to_drift(self, drift_type, model_performance_history=None):
        """Respond to detected drift."""
        response = {
            'action': None,
            'alert': False,
            'retrain': False,
            'weight_redistribution': False
        }
        
        if drift_type == 'sudden':
            # Sudden drift: immediate response needed
            response['action'] = 'partial_retraining'
            response['retrain'] = True
            response['alert'] = True
            response['weight_redistribution'] = True
            
        elif drift_type == 'gradual':
            # Gradual drift: monitor and prepare for retraining
            if model_performance_history:
                recent_perf = np.mean(model_performance_history[-10:])
                baseline_perf = np.mean(model_performance_history[:10])
                
                if recent_perf < baseline_perf * (1 - self.retraining_threshold):
                    response['action'] = 'partial_retraining'
                    response['retrain'] = True
                
                if recent_perf < baseline_perf * (1 - self.alert_threshold):
                    response['alert'] = True
            
            response['weight_redistribution'] = True
            
        elif drift_type == 'recurring':
            # Recurring drift: adaptive strategy
            response['action'] = 'adaptive_weighting'
            response['weight_redistribution'] = True
            response['alert'] = True
        
        return response
    
    def generate_alert(self, drift_info):
        """Generate alert message."""
        alert = {
            'severity': 'medium',
            'message': f"Concept drift detected: {drift_info['drift_type']}",
            'timestamp': pd.Timestamp.now(),
            'recommendations': []
        }
        
        if drift_info['drift_type'] == 'sudden':
            alert['severity'] = 'high'
            alert['recommendations'].append('Immediate model retraining recommended')
            alert['recommendations'].append('Review recent data for anomalies')
        
        elif drift_info['drift_type'] == 'gradual':
            alert['severity'] = 'medium'
            alert['recommendations'].append('Monitor performance trends')
            alert['recommendations'].append('Prepare for incremental retraining')
        
        elif drift_info['drift_type'] == 'recurring':
            alert['severity'] = 'high'
            alert['recommendations'].append('Implement adaptive ensemble weighting')
            alert['recommendations'].append('Consider seasonal/cyclical patterns')
        
        return alert


if __name__ == "__main__":
    # Example usage + artifact generation for the dashboard
    print("Testing Concept Drift Detection...")
    
    # Create synthetic data with drift
    np.random.seed(42)
    n_samples = 1000
    
    # First half: normal distribution
    data1 = np.random.normal(70, 10, n_samples // 2)
    
    # Second half: shifted distribution (drift)
    data2 = np.random.normal(60, 10, n_samples // 2)
    
    data = np.concatenate([data1, data2])
    
    # Initialize detector
    detector = ConceptDriftDetector()
    
    # Detect drift
    drift_count = 0
    for i, value in enumerate(data):
        drift_detected, adwin_drift, ph_drift = detector.detect_drift(value)
        if drift_detected:
            drift_count += 1
            if drift_count <= 3:  # Print first few drifts
                print(f"\nDrift detected at index {i}:")
                print(f"  Value: {value:.2f}")
                print(f"  ADWIN: {adwin_drift}")
                print(f"  Page-Hinkley: {ph_drift}")
                print(f"  Type: {detector.drift_history[-1]['drift_type']}")
    
    # Get summary
    summary = detector.get_drift_summary()
    if summary is not None:
        print(f"\nDrift Detection Summary:")
        print(f"Total drifts detected: {summary['total_drifts']}")
        print(f"Drift types: {summary['drift_types']}")
        print(f"ADWIN detections: {summary['adwin_detections']}")
        print(f"Page-Hinkley detections: {summary['ph_detections']}")
    else:
        print("\nNo drifts detected.")
        summary = {
            'total_drifts': 0,
            'drift_types': {},
            'adwin_detections': 0,
            'ph_detections': 0
        }
    
    # Persist artifacts for the Streamlit dashboard
    try:
        # Drift history as a flat table
        history_df = pd.DataFrame(detector.drift_history)
        history_df.to_csv("concept_drift_history.csv", index=False)
        
        # Summary (flatten drift_types dict to separate columns)
        flat_summary = {
            'total_drifts': summary.get('total_drifts', 0),
            'adwin_detections': summary.get('adwin_detections', 0),
            'ph_detections': summary.get('ph_detections', 0),
        }
        if isinstance(summary.get('drift_types'), dict):
            for k, v in summary['drift_types'].items():
                flat_summary[f"drift_type_{k}"] = v
        
        pd.DataFrame([flat_summary]).to_csv("concept_drift_summary.csv", index=False)
        print("\nConcept drift artifacts saved to 'concept_drift_history.csv' and 'concept_drift_summary.csv'.")
    except Exception as e:
        print(f"\nWarning: Failed to write concept drift artifacts: {e}")
    
    # Test drift response system
    response_system = DriftResponseSystem()
    
    print("\nTesting Drift Response System:")
    for drift_info in detector.drift_history[:3]:
        response = response_system.respond_to_drift(drift_info['drift_type'])
        alert = response_system.generate_alert(drift_info)
        
        print(f"\nDrift Type: {drift_info['drift_type']}")
        print(f"Response Action: {response['action']}")
        print(f"Alert: {alert['message']}")
        print(f"Severity: {alert['severity']}")
    
    print("\n" + "="*50)
    print("Module 6: Concept Drift Detection Complete!")
    print("="*50)
