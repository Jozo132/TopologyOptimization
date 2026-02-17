// Workflow step management — shows one step at a time with back/next buttons
export class WorkflowManager {
    constructor() {
        this.currentStep = 1;
        this.maxReachedStep = 1;
        this.steps = [1, 2, 3, 4];
        this.onStepChange = null; // callback(stepNumber)
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
                    this.goToStep(stepNum - 1);
                });
                nav.appendChild(backBtn);
            }

            if (stepNum < 4) {
                const nextBtn = document.createElement('button');
                nextBtn.className = 'btn-primary step-nav-next';
                nextBtn.textContent = 'Next →';
                nextBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.goToStep(stepNum + 1);
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

        // Disable next button if the next step hasn't been reached yet
        const currentEl = document.querySelector(`[data-step="${stepNumber}"]`);
        if (currentEl) {
            const nextBtn = currentEl.querySelector('.step-nav-next');
            if (nextBtn) {
                nextBtn.disabled = (stepNumber >= this.maxReachedStep);
            }
        }

        this.currentStep = stepNumber;
        if (this.onStepChange) this.onStepChange(stepNumber);
    }

    goToStep(stepNumber) {
        if (stepNumber < 1 || stepNumber > 4) return;
        if (stepNumber > this.maxReachedStep) return;
        this.showStep(stepNumber);
    }

    enableStep(stepNumber) {
        if (stepNumber > this.maxReachedStep) {
            this.maxReachedStep = stepNumber;
        }
        this.showStep(stepNumber);
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
