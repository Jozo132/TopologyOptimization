// Workflow step management
export class WorkflowManager {
    constructor() {
        this.currentStep = 1;
        this.steps = [1, 2, 3, 4];
    }

    enableStep(stepNumber) {
        const step = document.querySelector(`[data-step="${stepNumber}"]`);
        if (step) {
            step.removeAttribute('disabled');
            step.classList.add('active');
        }
        
        // Remove active from previous steps
        for (let i = 1; i < stepNumber; i++) {
            const prevStep = document.querySelector(`[data-step="${i}"]`);
            if (prevStep) {
                prevStep.classList.remove('active');
            }
        }
        
        this.currentStep = stepNumber;
    }

    disableStep(stepNumber) {
        const step = document.querySelector(`[data-step="${stepNumber}"]`);
        if (step) {
            step.setAttribute('disabled', 'true');
            step.classList.remove('active');
        }
    }

    reset() {
        // Disable all steps except the first
        this.steps.forEach(stepNum => {
            if (stepNum === 1) {
                this.enableStep(1);
            } else {
                this.disableStep(stepNum);
            }
        });
        this.currentStep = 1;
    }

    getCurrentStep() {
        return this.currentStep;
    }
}
