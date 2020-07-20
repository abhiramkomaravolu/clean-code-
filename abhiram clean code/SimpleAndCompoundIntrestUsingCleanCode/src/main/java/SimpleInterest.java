
public class SimpleInterest {
protected double principleAmount,annualInterestRate,Time;
	
	protected SimpleInterest(double principle,double InterestRate,double time)
	{
		this.principleAmount=principle;
		this.annualInterestRate=InterestRate/100;
		this.Time=time;
	}
	
	public double getSimpleInterest()
	{
		double finalAmount;
		
		finalAmount = principleAmount*(1+(annualInterestRate)*(Time));
		
		return finalAmount;
	}
}
