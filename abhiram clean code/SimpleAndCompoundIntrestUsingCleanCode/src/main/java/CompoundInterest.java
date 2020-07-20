
public class CompoundInterest {
protected double principleAmount,annualInterestRate,Time,numberOfPeriods;
	
	CompoundInterest(double principle,double InterestRate,double time,double nPeriods)
	{
		
		this.principleAmount=principle;
		this.annualInterestRate=InterestRate/100;
		this.Time=time;
		this.numberOfPeriods =nPeriods;
		
	}
	
	public double getCompoundInterest()
	{
		double finalAmount;
		
		finalAmount = principleAmount*Math.pow((1+((annualInterestRate)/numberOfPeriods)),(numberOfPeriods*Time));
		
		return finalAmount;
	}
}
