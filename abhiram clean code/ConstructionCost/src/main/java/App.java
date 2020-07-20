import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.*;

public class App 
{
    public static void main( String[] args )
    {   
        int MenuOption;
    	Scanner ToScanner= new Scanner(System.in);
		PrintStream Print=new PrintStream(new FileOutputStream(FileDescriptor.out));
		Print.println();
		do {
		Print.println("CONSTRUCTION COST ESTIMATION \n");
		Print.println("Variants Available");
		Print.println("1. Construction with standard materials \n2. Construction with above standard materials ");
		Print.println("3. Construction with high standard materials \n4. Construction with high standard materials and fully automated");
		Print.println("Enter your choice: ");
		MenuOption=ToScanner.nextInt();
    	}while(MenuOption>4);
		Print.print("enter the area of house");
		double Area=ToScanner.nextDouble();
		ConstructionCostEstimation ConstructionCostEstimationObject =new ConstructionCostEstimation(MenuOption,Area);
     Print.println("the cost estimated is" +ConstructionCostEstimationObject.getTotalCost()); 
	
     ToScanner.close();
     Print.close();
    }
}