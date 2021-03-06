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
        double[] arr1 = {154.895488,296.176064,440.460832,583.743072,728.009024,872.256544,1018.550112,1160.83936,1310.07792,1455.338144,1601.59856,1745.865024,1892.17264,2034.48144,2176.795488,2319.081248,2466.315744,2613.6456,2757.856128,2900.137376,3040.445696,3182.693696,3326.004256,3468.272992,3614.534144};
        double[] arr2 = {164.87,314.40,468.04,614.58,767.20,915.82,1069.45,1224.08,1376.61,1528.23,1679.89,1828.46,1981.08,2130.71,2278.30,2430.90,2579.54,2733.13,2883.73,3036.26,
                3187.88,3337.46,3487.09,3633.54,3782.12};
        double[] arr3 = {70.59,327.38,488.12,645.81,804.54,962.03,1119.71,1277.40,1433.16,1591.65,1749.37,1904.11,2064.79,2219.56,2374.13,2532.92,2691.61,2847.10,3000.83,3157.49,
                3315.23,3472.95,3628.64,3783.36,3940.04};
        double[] arr = {164.06,318.53,474.96,629.50,784.96,940.46,1096.91,1253.36,1408.84,1566.31,1724.83,1882.33,2037.76,2196.32,2352.87,2510.37,2667.69,2821.18,2980.76,3134.13,
                3289.55,3448.06,3606.51,3763.02,3921.44};
        double[] tmp = new double[arr.length];
        double avg = 156.85, res=0;        
        for ( int i = arr.length-1; i > 0 ; i-- ) {
            res += Math.pow(avg - (arr[i] - arr[i-1]), 2);
        }
        System.out.println(res/25);
    }
    
    void stdDeviationCf() {
        double[] arr1 = {209.53,380.83,558.11,746.44,921.73,1104.02,1286.33,1464.68,1634.02,1799.31,1983.65,2149.99,2332.32,2495.65,2674.98,2856.26,3020.57,3206.90,3379.18,3549.46,
                3727.81,3909.14,4078.44,4256.79,4440.08};
        double[] arr2 = {206.96,376.18,541.42,703.62,869.81,1058.99,1226.20,1406.45,1584.69,1754.91,1921.13,2098.34,2269.53,2441.73,2626.92,2796.12,2972.31,3140.51,3315.67,3482.89,
                3656.11,3830.36,4004.98,4168.21,4341.43};
        double[] arr3 = {208.40,383.05,557.69,721.33,894.98,1072.62,1260.57,1440.66,1599.29,1769.95,1937.64,2116.26,2298.89,2477.55,2664.21,2838.89,3008.54,3178.24,3349.88,3523.58,
                3696.21,3870.81,4061.56,4248.28,4419.98};
        double[] arr = {203.32,379.99,558.64,716.27,884.91,1051.52,1223.21,1403.84,1596.49,1770.13,1956.74,2142.40,2313.99,2483.67,2666.3,2842.94,3005.58,3178.20,3362.83,3537.49,
                3713.21,3887.84,4075.45,4252.12,4422.76};
        double[] tmp = new double[arr.length];
        double avg = 179.67, res=0;        
        for ( int i = arr.length-1; i > 0 ; i-- ) {
            res += Math.pow(avg - (arr[i] - arr[i-1]), 2);
        }
        res = res/25;
        res = Math.abs(Math.sqrt(res));
        System.out.println(res);
    }
    
    void stdDeviationBf() {
        double[] arr1 = {160.13,304.71,451.29,611.93,718.58,858.20,1003.81,1159.43,1307.04,1443.66,1615.33,1736.97,1872.56,2041.17,2172.78,2324.39,2470.02,2589.68,2721.30,2866.91,
                3027.54,3173.17,3315.78,3448.35,3581.0};
        double[] arr2 = {171.0,264.39,430.75,584.18,746.04,894.43,1044.84,1211.38,1377.78,1542.21,1694.68,1844.07,1995.47,2126.89,2289.30,2467.71,2639.14,2795.59,2976.02,3146.43,3285.89,3406.32,3531.70,3699.08,3852.53};
        double[] arr3 = {186.24,323.91,469.59,628.23,786.96,960.67,1110.36,1277.08,1466.76,1598.47,1756.18,1923.90,2090.60,2224.25,2389.92,2530.61,2674.27,2816.0,2985.72,3121.41,
                3266.08,3435.84,3580.57,3726.28,3886.96};
        double[] arr = {74.22,292.87,446.50,592.16,728.84,879.51,1023.16,1158.82,1304.52,1446.22,1604.94,1754.65,1896.33,2046.01,2205.69,2345.35,2484.0,2636.71,2781.46,2934.13,3077.81,3234.50,3377.20,3524.86,3661.54};
        double[] tmp = new double[arr.length];
        double avg = 175, res=0;        
        for ( int i = arr.length-1; i > 0 ; i-- ) {
            res += Math.pow(avg - (arr[i] - arr[i-1]), 2);
        }
        res = res/25;
        res = Math.abs(Math.sqrt(res));
        System.out.println(res);
    }
    
    public static void main(String args[]) {
        MeanCalculator obj = new MeanCalculator();
        obj.stdDeviationCf();
    }
}
