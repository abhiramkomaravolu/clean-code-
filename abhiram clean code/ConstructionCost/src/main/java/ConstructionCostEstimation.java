
public class ConstructionCostEstimation {
	protected double Area;
	protected int Index;
	ConstructionCostEstimation(int MenuOption,double Area) {
		this.Index =MenuOption;
		this.Area=Area;
		
		
	}
	public double getTotalCost() {
		int Cost[]= {1200,1500,1800,2500};
		return Cost[Index-1]*Area;
	}

}