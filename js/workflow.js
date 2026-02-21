// Workflow step management — shows one step at a time with back/next buttons
export class WorkflowManager {
    constructor() {
        this.currentStep = 1;
        this.maxReachedStep = 1;
        this.steps = [1, 2, 3, 4, 5, 6];
        this.totalSteps = 6;
        this.onStepChange = null; // callback(stepNumber)
    }

    /** Check if a step element is marked as skipped (e.g. manufacturing for non-topology) */
    _isStepHidden(stepNum) {
        const el = document.querySelector(`[data-step="${stepNum}"]`);
        if (!el) return true;
        return el.dataset.skip === 'true';
    }

    /** Find the next visible step after the given step */
    _nextVisibleStep(fromStep) {
        for (let s = fromStep + 1; s <= this.totalSteps; s++) {
            if (!this._isStepHidden(s)) return s;
        }
        return null;
    }

    /** Find the previous visible step before the given step */
    _prevVisibleStep(fromStep) {
        for (let s = fromStep - 1; s >= 1; s--) {
            if (!this._isStepHidden(s)) return s;
        }
        return null;
    }

    init() {
        // Insert navigation buttons into each step
        this.steps.forEach(stepNum => {
            const stepEl = document.querySelector(`[data-step="${stepNum}"]`);
            if (!stepEl) return;

            const nav = document.createElement('div');
            nav.className = 'step-nav';

            if (stepNum > 1) {
                const backBtn = document.createElement('button');
                backBtn.className = 'btn-secondary step-nav-back';
                backBtn.textContent = '← Back';
                backBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const prev = this._prevVisibleStep(stepNum);
                    if (prev !== null) this.goToStep(prev);
                });
                nav.appendChild(backBtn);
            }

            if (stepNum < this.totalSteps) {
                const nextBtn = document.createElement('button');
                nextBtn.className = 'btn-primary step-nav-next';
                nextBtn.textContent = 'Next →';
                nextBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const next = this._nextVisibleStep(stepNum);
                    if (next !== null) this.goToStep(next);
                });
                nav.appendChild(nextBtn);
            }

            const content = stepEl.querySelector('.step-content');
            if (content) content.appendChild(nav);
        });

        this.showStep(1);
    }

    /** Show a specific step and hide all others */
    showStep(stepNumber) {
        this.steps.forEach(num => {
            const el = document.querySelector(`[data-step="${num}"]`);
            if (!el) return;
            if (num === stepNumber) {
                el.style.display = '';
                el.removeAttribute('disabled');
                el.classList.add('active');
            } else {
                el.style.display = 'none';
                el.classList.remove('active');
            }
        });

        // Update next button state
        const currentEl = document.querySelector(`[data-step="${stepNumber}"]`);
        if (currentEl) {
            const nextBtn = currentEl.querySelector('.step-nav-next');
            if (nextBtn) {
                nextBtn.disabled = false;
            }
        }

        this.currentStep = stepNumber;
        if (this.onStepChange) this.onStepChange(stepNumber);
    }

    goToStep(stepNumber) {
        if (stepNumber < 1 || stepNumber > this.totalSteps) return;
        // Allow advancing to the immediate next step from current
        if (stepNumber === this.currentStep + 1 && stepNumber === this.maxReachedStep + 1) {
            this.maxReachedStep = stepNumber;
        }
        // Also allow skipping hidden steps (e.g. manufacturing when non-topology)
        if (stepNumber > this.maxReachedStep + 1) {
            // Check if all steps between maxReached+1 and stepNumber are hidden
            let allHidden = true;
            for (let s = this.maxReachedStep + 1; s < stepNumber; s++) {
                if (!this._isStepHidden(s)) { allHidden = false; break; }
            }
            if (allHidden && stepNumber === this.maxReachedStep + 2) {
                this.maxReachedStep = stepNumber;
            }
        }
        if (stepNumber > this.maxReachedStep) return;
        this.showStep(stepNumber);
    }

    enableStep(stepNumber) {
        if (stepNumber > this.maxReachedStep) {
            this.maxReachedStep = stepNumber;
        }
    }

    disableStep(stepNumber) {
        // If we're disabling a step above current max, do nothing
        if (stepNumber <= this.maxReachedStep && stepNumber > 1) {
            // Only reduce maxReached if disabling the highest step
            if (stepNumber === this.maxReachedStep) {
                this.maxReachedStep = stepNumber - 1;
            }
        }
    }

    reset() {
        this.maxReachedStep = 1;
        this.showStep(1);
    }

    getCurrentStep() {
        return this.currentStep;
    }
}
