	import java.io.FileDescriptor;
	import java.io.FileOutputStream;
	import java.io.PrintStream;
	import java.util.Scanner;

	public class InterestCalculation
	{
	    public static void main( String[] args )
	    {
	    	 double principle, rate, timePeriod, periodDivision;

	         Scanner scanInput = new Scanner(System.in);
	         PrintStream printOutput=new PrintStream(new FileOutputStream(FileDescriptor.out));
	         int menuOption;
	         do {
	         printOutput.println("Menu");
	         printOutput.println("1.Simple intrest ");
	         printOutput.println("2.compound intrest");
	         menuOption = scanInput.nextInt();
	         }while(menuOption>2);
	         printOutput.println("1. Principle Amount: ");
	         principle = scanInput.nextDouble();

	         printOutput.println("2. Rate of Interest: ");
	         rate = scanInput.nextDouble();

	         printOutput.println("3. No. of Years for which interest is to be calculated: ");
	         timePeriod = scanInput.nextDouble();
             if(menuOption==1) { 
            	 SimpleInterest loanSimpleInterest = new SimpleInterest(principle, rate, timePeriod);
            	  printOutput.println("for Simple Interest: " + loanSimpleInterest.getSimpleInterest());
            	
                               }
            else {
            	 printOutput.println("4.No.of times for which Compound Interest is calculated Every year: ");
    	         periodDivision = scanInput.nextDouble();
    	         CompoundInterest loanCompoundInterest = new CompoundInterest(principle, rate, timePeriod, periodDivision);
    	         printOutput.println("for Compound Interest: " + loanCompoundInterest.getCompoundInterest());
                }
	        printOutput.close();
	         scanInput.close();
	     }
	}



