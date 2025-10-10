print("6. static/js/main.js")
print("-" * 80)
main_js = """// Main JavaScript file for general functionality

document.addEventListener('DOMContentLoaded', function() {
    // Auto-hide flash messages after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    });

    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Form validation enhancement
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const requiredFields = form.querySelectorAll('[required]');
            let isValid = true;

            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    field.style.borderColor = 'red';
                } else {
                    field.style.borderColor = '';
                }
            });

            if (!isValid) {
                e.preventDefault();
                alert('Please fill in all required fields');
            }
        });
    });

    // Password confirmation validation
    const passwordForm = document.querySelector('form[action*="signup"]');
    if (passwordForm) {
        passwordForm.addEventListener('submit', function(e) {
            const password = document.getElementById('password');
            const confirmPassword = document.getElementById('confirm_password');
            
            if (password && confirmPassword) {
                if (password.value !== confirmPassword.value) {
                    e.preventDefault();
                    alert('Passwords do not match!');
                    confirmPassword.style.borderColor = 'red';
                    return false;
                }
            }
        });
    }

    // Add loading spinner on form submission
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '⏳ Processing...';
            }
        });
    });
});

// Utility function to show notifications
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type}`;
    notification.textContent = message;
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.minWidth = '300px';
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Confirmation dialog helper
function confirmAction(message) {
    return confirm(message);
}
"""
print(main_js)
print("\n")

# 7. static/js/prediction.js
print("7. static/js/prediction.js")
print("-" * 80)
prediction_js = """// Prediction page specific JavaScript

document.addEventListener('DOMContentLoaded', function() {
    
    // Symptom checkbox interactions
    const symptomCheckboxes = document.querySelectorAll('.symptom-checkbox input[type="checkbox"]');
    let selectedCount = 0;

    symptomCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            // Update selected count
            selectedCount = document.querySelectorAll('.symptom-checkbox input:checked').length;
            updateSelectedCount();
            
            // Add visual feedback
            const label = this.closest('.symptom-checkbox');
            if (this.checked) {
                label.style.backgroundColor = '#e8f0fe';
                label.style.borderColor = '#667eea';
            } else {
                label.style.backgroundColor = '';
                label.style.borderColor = '';
            }
        });
    });

    // Display selected symptoms count
    function updateSelectedCount() {
        let countDisplay = document.getElementById('selectedCount');
        if (!countDisplay) {
            countDisplay = document.createElement('div');
            countDisplay.id = 'selectedCount';
            countDisplay.style.position = 'sticky';
            countDisplay.style.top = '20px';
            countDisplay.style.padding = '15px';
            countDisplay.style.backgroundColor = '#667eea';
            countDisplay.style.color = 'white';
            countDisplay.style.borderRadius = '8px';
            countDisplay.style.textAlign = 'center';
            countDisplay.style.fontWeight = 'bold';
            countDisplay.style.marginBottom = '20px';
            
            const symptomsSection = document.querySelector('.symptoms-section');
            if (symptomsSection) {
                symptomsSection.insertBefore(countDisplay, symptomsSection.querySelector('.symptoms-grid'));
            }
        }
        
        countDisplay.textContent = `✓ ${selectedCount} symptoms selected`;
        
        if (selectedCount === 0) {
            countDisplay.style.backgroundColor = '#999';
        } else if (selectedCount <= 2) {
            countDisplay.style.backgroundColor = '#10b981';
        } else if (selectedCount <= 4) {
            countDisplay.style.backgroundColor = '#f59e0b';
        } else {
            countDisplay.style.backgroundColor = '#dc2626';
        }
    }

    // Search/filter symptoms
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = '🔍 Search symptoms...';
    searchInput.style.width = '100%';
    searchInput.style.padding = '12px';
    searchInput.style.marginBottom = '20px';
    searchInput.style.border = '2px solid #e0e0e0';
    searchInput.style.borderRadius = '8px';
    searchInput.style.fontSize = '16px';

    const symptomsGrid = document.querySelector('.symptoms-grid');
    if (symptomsGrid) {
        symptomsGrid.parentNode.insertBefore(searchInput, symptomsGrid);
    }

    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const symptoms = document.querySelectorAll('.symptom-checkbox');
        
        symptoms.forEach(symptom => {
            const text = symptom.textContent.toLowerCase();
            if (text.includes(searchTerm)) {
                symptom.style.display = 'flex';
            } else {
                symptom.style.display = 'none';
            }
        });
    });

    // Quick select buttons
    const quickSelectDiv = document.createElement('div');
    quickSelectDiv.style.marginBottom = '20px';
    quickSelectDiv.style.display = 'flex';
    quickSelectDiv.style.gap = '10px';
    quickSelectDiv.innerHTML = `
        <button type="button" id="selectAllBtn" class="btn-small" style="padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">
            Select All
        </button>
        <button type="button" id="clearAllBtn" class="btn-small" style="padding: 8px 16px; background: #6b7280; color: white; border: none; border-radius: 5px; cursor: pointer;">
            Clear All
        </button>
    `;

    if (symptomsGrid) {
        symptomsGrid.parentNode.insertBefore(quickSelectDiv, symptomsGrid);
    }

    // Select All functionality
    document.getElementById('selectAllBtn')?.addEventListener('click', function() {
        symptomCheckboxes.forEach(checkbox => {
            if (!checkbox.checked) {
                checkbox.checked = true;
                checkbox.dispatchEvent(new Event('change'));
            }
        });
    });

    // Clear All functionality
    document.getElementById('clearAllBtn')?.addEventListener('click', function() {
        symptomCheckboxes.forEach(checkbox => {
            if (checkbox.checked) {
                checkbox.checked = false;
                checkbox.dispatchEvent(new Event('change'));
            }
        });
    });

    // Form validation before submission
    const predictionForm = document.querySelector('.prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            const checkedSymptoms = document.querySelectorAll('.symptom-checkbox input:checked');
            const customIssue = document.querySelector('textarea[name="custom_issue"]');
            
            if (checkedSymptoms.length === 0 && (!customIssue || !customIssue.value.trim())) {
                e.preventDefault();
                alert('⚠️ Please select at least one symptom or describe your issue in the text area.');
                return false;
            }
            
            // Show loading state
            const submitBtn = predictionForm.querySelector('.btn-submit');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '🔄 Analyzing Symptoms...';
            }
        });
    }

    // Character counter for custom issue textarea
    const customIssueTextarea = document.querySelector('textarea[name="custom_issue"]');
    if (customIssueTextarea) {
        const charCounter = document.createElement('div');
        charCounter.style.textAlign = 'right';
        charCounter.style.fontSize = '14px';
        charCounter.style.color = '#666';
        charCounter.style.marginTop = '5px';
        
        customIssueTextarea.parentNode.appendChild(charCounter);
        
        customIssueTextarea.addEventListener('input', function() {
            const length = this.value.length;
            charCounter.textContent = `${length} characters`;
            
            if (length > 500) {
                charCounter.style.color = '#dc2626';
            } else {
                charCounter.style.color = '#666';
            }
        });
        
        // Trigger initial update
        customIssueTextarea.dispatchEvent(new Event('input'));
    }

    // Initialize count
    updateSelectedCount();
});

// Common symptom presets
const commonPresets = {
    'cold': ['fever', 'cough', 'sore_throat', 'runny_nose', 'fatigue'],
    'flu': ['fever', 'body_ache', 'chills', 'fatigue', 'headache'],
    'stomach': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain']
};

function applyPreset(presetName) {
    const symptoms = commonPresets[presetName];
    if (symptoms) {
        // Clear all first
        document.querySelectorAll('.symptom-checkbox input').forEach(cb => {
            cb.checked = false;
        });
        
        // Select preset symptoms
        symptoms.forEach(symptom => {
            const checkbox = document.querySelector(`input[name="${symptom}"]`);
            if (checkbox) {
                checkbox.checked = true;
                checkbox.dispatchEvent(new Event('change'));
            }
        });
    }
}
"""
print(prediction_js)
print("\n")