import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

public class MeanCalculator {
    
    String fileName;
    
    MeanCalculator() {
        fileName = null;
    }
    
    /**
     * Takes file name as argument and returns gpu time
     * @param fileName
     * @return
     */
    double gpuTime(String fileName) {
        File csv = null;
        BufferedReader br = null;
        String line = null;
        String[] arr = null;
        double  totalGpuTime = 0;
        
        try {
            csv = new File(fileName);
            br = new BufferedReader(new FileReader(fileName));
            br.readLine();br.readLine();br.readLine();br.readLine();br.readLine();br.readLine();
            while((line = br.readLine())!= null) {
                arr = line.split(",");
                totalGpuTime += Double.parseDouble(arr[1]);                
            }
            br.close();           
        } catch (Exception e) {
            e.printStackTrace();
        }        
        return totalGpuTime;
    }
    
    void stdDeviationLacf() {
        double[] arr = {154.895488,296.176064,440.460832,583.743072,728.009024,872.256544,1018.550112,1160.83936,1310.07792,1455.338144,1601.59856,1745.865024,1892.17264,2034.48144,2176.795488,2319.081248,2466.315744,2613.6456,2757.856128,2900.137376,3040.445696,3182.693696,3326.004256,3468.272992,3614.534144};
        double[] tmp = new double[arr.length];
        double avg = 144.58, res=0;        
        for ( int i = arr.length-1; i > 0 ; i-- ) {
            res += Math.pow(avg - (arr[i] - arr[i-1]), 2);
        }
        System.out.println(res/25);
    }
    
    public static void main(String args[]) {
        MeanCalculator obj = new MeanCalculator();
//        obj.gpuTime("cuda_profile_0.csv");
        
    }
}
