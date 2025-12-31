"""
Linear Programming Optimization Module
Generates personalized 30-day savings plans
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import logging
from typing import Dict, Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SavingsPlanOptimizer:
    """
    Linear Programming optimizer for personalized 30-day savings plans
    """
    
    def __init__(self, days: int = 30):
        """
        Initialize optimizer
        
        Args:
            days: Number of days for planning
        """
        self.days = days
        self.results = None
        
    def create_optimization_problem(self, daily_income: float,
                                   avg_fixed_expenses: float,
                                   avg_variable_expenses: float,
                                   target_savings_ratio: float = 0.2) -> Dict:
        """
        Set up linear programming problem for savings optimization
        
        Objective: Maximize total savings over 30 days
        Constraints:
            - Daily spending <= Daily income
            - Fixed expenses >= minimum fixed expenses
            - Variable expenses <= adjustable limit
            - Total savings >= Target savings goal
        
        Args:
            daily_income: Average daily income
            avg_fixed_expenses: Average daily fixed expenses
            avg_variable_expenses: Average daily variable expenses
            target_savings_ratio: Target ratio of savings to income
            
        Returns:
            Optimization results dictionary
        """
        # Decision variables for each day:
        # [daily_savings_0, ..., daily_savings_29,
        #  daily_variable_expenses_0, ..., daily_variable_expenses_29]
        
        n_vars = self.days * 2  # savings and variable expenses for each day
        
        # Objective: Maximize total savings (minimize negative savings)
        # We want to maximize sum(savings), so we minimize -sum(savings)
        c = np.zeros(n_vars)
        c[:self.days] = -1  # Negative because linprog minimizes
        
        # Inequality constraints (A_ub @ x <= b_ub)
        A_ub = []
        b_ub = []
        
        # Constraint 1: Each day's spending <= daily income
        # fixed_expenses + variable_expenses_day <= daily_income
        for day in range(self.days):
            constraint = np.zeros(n_vars)
            constraint[self.days + day] = 1  # variable expense for this day
            A_ub.append(constraint)
            b_ub.append(daily_income - avg_fixed_expenses)
        
        # Constraint 2: Variable expenses >= 0
        for day in range(self.days):
            constraint = np.zeros(n_vars)
            constraint[self.days + day] = -1
            A_ub.append(constraint)
            b_ub.append(0)
        
        # Constraint 3: Daily savings <= daily surplus
        for day in range(self.days):
            constraint = np.zeros(n_vars)
            constraint[day] = 1  # savings
            constraint[self.days + day] = 1  # variable expenses
            A_ub.append(constraint)
            b_ub.append(daily_income - avg_fixed_expenses)
        
        # Constraint 4: Variable expenses <= average with flexibility
        max_variable = avg_variable_expenses * 1.5  # Allow 50% flexibility
        for day in range(self.days):
            constraint = np.zeros(n_vars)
            constraint[self.days + day] = 1
            A_ub.append(constraint)
            b_ub.append(max_variable)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Equality constraints (A_eq @ x = b_eq)
        # Total savings = daily savings sum
        A_eq = None
        b_eq = None
        
        # Bounds: all variables >= 0
        bounds = [(0, None) for _ in range(n_vars)]
        
        logger.info(f"LP problem created with {n_vars} variables and {len(A_ub)} constraints")
        
        return {
            'c': c,
            'A_ub': A_ub,
            'b_ub': b_ub,
            'A_eq': A_eq,
            'b_eq': b_eq,
            'bounds': bounds,
            'daily_income': daily_income,
            'avg_fixed': avg_fixed_expenses,
            'avg_variable': avg_variable_expenses
        }
    
    def solve_optimization(self, problem: Dict) -> Tuple[bool, np.ndarray]:
        """
        Solve the linear programming problem
        
        Args:
            problem: Problem dictionary from create_optimization_problem
            
        Returns:
            Success flag and solution array
        """
        result = linprog(
            c=problem['c'],
            A_ub=problem['A_ub'],
            b_ub=problem['b_ub'],
            A_eq=problem['A_eq'],
            b_eq=problem['b_eq'],
            bounds=problem['bounds'],
            method='highs'
        )
        
        if result.success:
            logger.info(f"Optimization successful! Maximum savings: ${-result.fun:.2f}")
        else:
            logger.warning(f"Optimization failed: {result.message}")
        
        return result.success, result.x if result.success else None
    
    def generate_savings_plan(self, daily_income: float,
                            avg_fixed_expenses: float,
                            avg_variable_expenses: float,
                            behavioral_risk_score: float = 0.5,
                            behavioral_persona: int = 0) -> pd.DataFrame:
        """
        Generate personalized 30-day savings plan
        
        Args:
            daily_income: Average daily income
            avg_fixed_expenses: Average daily fixed expenses
            avg_variable_expenses: Average daily variable expenses
            behavioral_risk_score: Risk score (0-1) to adjust recommendations
            behavioral_persona: Behavioral persona cluster
            
        Returns:
            DataFrame with daily recommendations
        """
        # Create optimization problem
        problem = self.create_optimization_problem(
            daily_income, avg_fixed_expenses, avg_variable_expenses
        )
        
        # Solve
        success, x = self.solve_optimization(problem)
        
        if not success or x is None:
            logger.warning("Optimization failed, using heuristic approach")
            x = self._heuristic_plan(daily_income, avg_fixed_expenses, avg_variable_expenses)
        
        # Extract solutions
        daily_savings = x[:self.days]
        daily_variable = x[self.days:self.days*2]
        
        # Create plan dataframe
        plan_df = pd.DataFrame({
            'day': range(1, self.days + 1),
            'daily_income': daily_income,
            'fixed_expenses': avg_fixed_expenses,
            'variable_expenses': daily_variable,
            'recommended_savings': daily_savings,
            'total_spending': avg_fixed_expenses + daily_variable,
            'behavioral_risk': [behavioral_risk_score] * self.days,
        })
        
        plan_df['spending_ratio'] = plan_df['total_spending'] / plan_df['daily_income']
        plan_df['cumulative_savings'] = plan_df['recommended_savings'].cumsum()
        
        # Personalize based on behavioral risk
        if behavioral_risk_score > 0.7:
            # High risk: more conservative recommendations
            plan_df['variable_expenses'] = plan_df['variable_expenses'] * 0.7
            plan_df['recommended_savings'] = plan_df['recommended_savings'] * 1.3
        elif behavioral_risk_score < 0.3:
            # Low risk: can afford more flexibility
            plan_df['variable_expenses'] = plan_df['variable_expenses'] * 1.1
        
        # Recalculate totals
        plan_df['total_spending'] = plan_df['fixed_expenses'] + plan_df['variable_expenses']
        plan_df['cumulative_savings'] = plan_df['recommended_savings'].cumsum()
        
        self.results = plan_df
        logger.info(f"Savings plan generated. Total 30-day savings: ${plan_df['recommended_savings'].sum():.2f}")
        
        return plan_df
    
    def _heuristic_plan(self, daily_income: float,
                       avg_fixed_expenses: float,
                       avg_variable_expenses: float) -> np.ndarray:
        """
        Heuristic approach when optimization fails
        
        Args:
            daily_income: Daily income
            avg_fixed_expenses: Fixed expenses
            avg_variable_expenses: Variable expenses
            
        Returns:
            Solution array
        """
        daily_surplus = daily_income - avg_fixed_expenses - avg_variable_expenses
        
        x = np.zeros(self.days * 2)
        
        # Savings: save 60% of surplus each day
        x[:self.days] = max(0, daily_surplus * 0.6)
        
        # Variable expenses: reduce by 20%
        x[self.days:] = avg_variable_expenses * 0.8
        
        return x
    
    def get_savings_summary(self) -> Dict:
        """
        Get summary statistics of the savings plan
        
        Returns:
            Summary dictionary
        """
        if self.results is None:
            return {}
        
        return {
            'total_savings_30_days': self.results['recommended_savings'].sum(),
            'average_daily_savings': self.results['recommended_savings'].mean(),
            'average_daily_spending': self.results['total_spending'].mean(),
            'total_fixed_expenses': self.results['fixed_expenses'].sum(),
            'total_variable_expenses': self.results['variable_expenses'].sum(),
            'average_spending_ratio': self.results['spending_ratio'].mean(),
            'max_daily_savings': self.results['recommended_savings'].max(),
            'min_daily_savings': self.results['recommended_savings'].min(),
        }
    
    def adjust_plan_by_event(self, plan_df: pd.DataFrame,
                            event_date: int,
                            event_type: str = 'emergency',
                            event_cost: float = 50) -> pd.DataFrame:
        """
        Adjust savings plan based on unexpected events
        
        Args:
            plan_df: Original plan DataFrame
            event_date: Day of event (1-30)
            event_type: Type of event ('emergency', 'opportunity', 'windfall')
            event_cost: Cost/benefit of event
            
        Returns:
            Adjusted plan DataFrame
        """
        adjusted_plan = plan_df.copy()
        
        if event_type == 'emergency':
            # Reduce savings on event day
            adjusted_plan.loc[event_date-1, 'recommended_savings'] = max(0,
                adjusted_plan.loc[event_date-1, 'recommended_savings'] - event_cost)
        
        elif event_type == 'windfall':
            # Add to savings on event day
            adjusted_plan.loc[event_date-1, 'recommended_savings'] += event_cost
        
        elif event_type == 'opportunity':
            # Allow additional spending, reduce future savings
            remaining_days = self.days - event_date
            if remaining_days > 0:
                daily_reduction = event_cost / remaining_days
                adjusted_plan.loc[event_date:, 'recommended_savings'] -= daily_reduction
        
        # Recalculate cumulative
        adjusted_plan['cumulative_savings'] = adjusted_plan['recommended_savings'].cumsum()
        
        return adjusted_plan
