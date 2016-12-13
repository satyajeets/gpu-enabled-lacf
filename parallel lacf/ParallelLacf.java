import edu.rit.pj2.Task;
import edu.rit.gpu.Kernel;
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuIntArray;
import edu.rit.gpu.GpuStructArray;
import edu.rit.gpu.Module;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.TreeSet;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import edu.rit.gpu.CacheConfig;

interface KernelFunctionInterface extends Kernel {
    public void parallelLookup(GpuStructArray<FilterEntry> entry);    
}

interface KernelFunctionInterface1 extends Kernel {
    public void parallelLookup1(GpuIntArray md5Ip,
                                GpuIntArray shaIp,
                                GpuIntArray md5Fp, 
                                GpuIntArray shaFp,
                                GpuIntArray result);
}

public class ParallelLacf extends Task {
    
    private static final int fsize = 4000;
    private static final int MAX_KICKS = 50;
    static int successCnt=0, failureCnt=0;
    FilterEntry[] filter;
    Gpu gpu;
    Module module;
    KernelFunctionInterface kernel;
    KernelFunctionInterface1 kernel1;
    HashMap<Integer, HashMap<Long, String>> routeTable;
    HashMap<Long, String> prefixTable;
    int hashProbe;
    static double time;
    
//    ParallelLacf() {
//        hashProbe = 0;
//        filter = new FilterEntry[fsize];
//        for(int i = 0; i < fsize; i++)
//            filter[i] = new FilterEntry();
//    }
    
    void init() {
        routeTable = new HashMap<>();//remove!
        hashProbe = 0;
        time = 0;        
        filter = new FilterEntry[fsize];
        for(int i = 0; i < fsize; i++)            
            filter[i] = new FilterEntry();        
    }
    
    /*** Helper methods section ***/
    
    public long ipToDecimal(String info) {
        
        String ip[] = info.split("\\.");
        
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < ip.length; i++) {
            sb.append(String.format("%02X", Integer.parseInt(ip[i])));  
        }
        String pre = sb.toString();
        Long prefix = Long.parseLong(pre, 16);
        return prefix;
    }
    
    public boolean findPopularity(int prefixLength) {
        if(prefixLength >= 14 && prefixLength <= 24) {
            return true;
        } else {
            return false;
        }
    }
    
    //converts byte array to hex string
    private static String convertByteArrayToHexString(byte[] arrayBytes) {
        StringBuffer stringBuffer = new StringBuffer();
        for (int i = 0; i < arrayBytes.length; i++) {
            stringBuffer.append(
                    Integer.toString((arrayBytes[i] & 0xFF) + 0x100, 16)
                    .substring(1));
        }
        return stringBuffer.toString();
    }
    
    //just computes md5 of string and returns hex value
    static String hashString(String message, String algorithm) {
     
        try {
            MessageDigest digest = MessageDigest.getInstance(algorithm);
            byte[] hashedBytes = digest.digest(message.getBytes("UTF-8"));
     
            return convertByteArrayToHexString(hashedBytes);
        } catch (NoSuchAlgorithmException | UnsupportedEncodingException ex) {
            System.out.println("ERROR in hashing");
            System.exit(0);
        }
        return null;
    }

   //generates fingerprint
    public char generateFingerprint(String prefix) {
        String fp = hashString(prefix, "MD5");        
        BigInteger val = new BigInteger(fp, 16);
        int mappedValue = val.intValue();        
        mappedValue = (mappedValue & 0xFF) % 127;
        char fingerprint = (char) mappedValue;
        return fingerprint;
    }
    
    //1st possible position in cuckoo filter
    public int getPosition1(String prefix, int option) {
        String hash;
        if(option == 1) {
            hash = hashString(prefix, "MD5");
        } else {
            hash = hashString(prefix, "SHA");
        }
        BigInteger val = new BigInteger(hash, 16);
        int mappedValue = val.intValue();
        mappedValue = mappedValue % fsize;
        if(mappedValue < 0) {
            mappedValue = mappedValue * -1;
        }
        return mappedValue;
    }
    
    //2nd possible position in cuckoo filter
    public int getPosition2(int pos1, char fp, int option) {
        String prefix = fp + "";
        String hash;
        if(option == 1) {
            hash = hashString(prefix, "MD5");
        } else {
            hash = hashString(prefix, "SHA");
        }
        BigInteger val = new BigInteger(hash, 16);
        int mappedValue = val.intValue();
        mappedValue = mappedValue % fsize;
        if(mappedValue < 0) {
            mappedValue = mappedValue * -1;
        }        
        int pos2 = mappedValue ^ pos1;
        pos2 = pos2 % fsize;
        return pos2;        
    }
    
    void displayResults(double totalTime) {
        System.out.println("Success count: " + successCnt);
        System.out.println("Failure count: " + failureCnt);        
        System.out.println("Hash probes: " + hashProbe);
        System.out.println("Final Avg : " + totalTime/25);
    }
    
    /*** Helper methods section ends ***/
    
    /*** Filter creation method section ***/
    
    public boolean addItem(char fingerprint, int pos1, int pos2,int option) {
        if(insertAtPos(fingerprint, pos1) == true) {        
            return true;
        }
        
        if(insertAtPos(fingerprint, pos2) == true) {
            return true;
        }
        
        int pos = pos1;
        
        for(int i = 0; i < MAX_KICKS; i++) {
            char temp = (char)filter[pos].fingerprint[4];
            filter[pos].fingerprint[4] = (byte)fingerprint;
            fingerprint = temp;
            
            pos = getPosition2(pos, fingerprint, option);
            
            if(insertAtPos(fingerprint, pos) == true) {
                return true;
            }
        }
        System.out.println("Max kicks exceeded");
        return false;
    }
    
    void createFilter(String prefixesFileName) throws Exception {
        File prefixesFile = new File(prefixesFileName);
        BufferedReader br  = new BufferedReader(new FileReader(prefixesFile));
        String line;
        
        while((line = br.readLine())!= null) {
            line = line.replace("\n", "");
            String info[] = line.split("/");
            int prefixLength = Integer.parseInt(info[1]);
            long prefix = ipToDecimal(info[0]);
            prefix = prefix >> (32 - prefixLength);
            //to be removed!
            if(routeTable.get(prefixLength) == null) {
                prefixTable = new HashMap<>();
                prefixTable.put(prefix, "nexthop");
                routeTable.put(prefixLength, prefixTable);
            } else {
                routeTable.get(prefixLength).put(prefix, "nexthop");
            }

//            int pos1 = -1;
//            int pos2 = -1;
//            int pos11 = -1;
//            int pos12 = -1;
//            char fingerprint = generateFingerprint(Long.toString(prefix));            
//            pos1 = getPosition1(Long.toString(prefix), 1);
//            pos2 = getPosition2(pos1, fingerprint, 1);
//            if ( findPopularity(prefixLength) == true ) {
//                pos11 = getPosition1(Long.toString(prefix), 2);
//                pos12 = getPosition2(pos11, fingerprint, 2);
//            }
//
//            addItem(fingerprint, pos1, pos2, 1);
//            //double insertion for non-popular prefix
//            if(pos11 != -1 && pos12 != -1) {
//                addItem(fingerprint, pos11, pos12, 2);
//            }
//
        }

        br.close();        
    }
    
    public boolean insertAtPos(char fp, int pos) {
        if(filter[pos].size == 5) {
            return false;
        } else if(filter[pos].size == 0) {
            filter[pos] = new FilterEntry();
            filter[pos].fingerprint[0] = (byte)fp;
            filter[pos].size += 1;
        } else {
            filter[pos] = new FilterEntry();
            filter[pos].fingerprint[filter[pos].size] = (byte)fp;
            filter[pos].size += 1;
        }
        return true;
    }
    
    /*** Filter creation section ends ***/
    
    /*** Filter lookup section begins ***/
    
    void lookupOperation(String fileName) throws Exception {        
        File testDataFile = new File(fileName);
        BufferedReader br  = new BufferedReader(new FileReader(testDataFile));
        String line;
        long ip;
        boolean found = false;//tracks if this Ip was found or not
        ArrayList<String> testData = new ArrayList<String>();        
        GpuIntArray md5Ip = gpu.getIntArray(33);
        GpuIntArray shaIp = gpu.getIntArray(33);
        GpuIntArray md5Fp = gpu.getIntArray(33);        
        GpuIntArray shaFp = gpu.getIntArray(33);
        GpuIntArray result = gpu.getIntArray(33);

        while((line = br.readLine())!= null) {
            line = line.replace("\n", "");
            testData.add(line);
        }
        br.close();
        Clock t1 = new Clock();
        for ( String l : testData ) {
            ip = ipToDecimal(l);
            for(int index = 32; index >= 1; index--) {
                //create MD5 arrays
                String fp = hashString(Long.toString(ip), "MD5");
                BigInteger val = new BigInteger(fp, 16);
                int mappedValue = val.intValue();                
                md5Ip.item[index] = mappedValue;
                mappedValue = (mappedValue & 0xFF) % 127;
                char fingerprint = (char) mappedValue;
                //md5 of fingerprint needed for pos2
                String prefix = fingerprint + "";
                String hash;
                hash = hashString(prefix, "MD5");                
                val = new BigInteger(hash, 16);
                md5Fp.item[index] = val.intValue();
                //sha of prefix/IP
                fp = hashString(Long.toString(ip), "SHA");
                val = new BigInteger(fp, 16);
                shaIp.item[index] = val.intValue();
                //sha of fingerprint needed for pos2
                hash = hashString(prefix, "SHA");
                val = new BigInteger(hash, 16);
                shaFp.item[index] = val.intValue();
                //shift
                ip = ip >> 1;
            }
            //parallel lookup. GPU call
            md5Ip.hostToDev();
            shaIp.hostToDev();
            md5Fp.hostToDev();
            shaIp.hostToDev();
         
            kernel1.parallelLookup1(md5Ip, shaIp, md5Fp, shaFp, result);

            result.devToHost();
            
            //start looking from highest
            TreeSet<Integer> possibleLens = new TreeSet<Integer>(Collections.reverseOrder());
            for ( int i = 1 ; i < 33 ; i++ ) {                
                if ( result.item[i] != -1 ) {
                    possibleLens.add(result.item[i]);
                }
            } 

            //now lookup hashtable/actual forwarding table
            Iterator<Integer> it = possibleLens.iterator();
            ip = ipToDecimal(l);
            t1.reset();
            t1.start();
            while ( it.hasNext() ) {
                hashProbe++;
                int index = (int)it.next();
                long tmpIp = ip >> (32 - index);
                if ( routeTable.get(index) != null ) {
                    if ( routeTable.get(index).get(tmpIp) != null ) {
                        //have found it!                        
                        successCnt++;
                        found = true;
                        break;
                    }
                }
            }
            t1.stop();
            time = time + t1.getTime();
            
            if ( found == false ) {                
                failureCnt++;
            }
            found = false;
        }        
    }
    
    void newCreateFilter() {
        CreateFilter createFilterObj = new CreateFilter();
        filter = new FilterEntry[fsize];
        createFilterObj.createFilter("p1.txt");
        for ( int i = 0 ; i < createFilterObj.filter.length ; i++ ) {
            filter[i] = new FilterEntry();
            for ( int j = 0 ; j < 5 ; j++ )
                filter[i].fingerprint[j] = createFilterObj.filter[i].fingerprint[j];
        }
    }

    /*** MAIN method ***/

    public void main(String[] arg0) throws Exception {        
        ParallelLacf obj = new ParallelLacf();
        boolean isFirstRun = true;        
        MeanCalculator mc = new MeanCalculator();
        double totalTime = 0, firstTime=0, firstGpu = 0;;
        obj.init();
        try {
            obj.createFilter("p1.txt");
            obj.newCreateFilter();
            obj.setupGPU();
            obj.moveFilterToGpu();
//            //find mean of 100 runs
            for ( int j = 0 ; j < 25 ; j++) {
                //one run = 10,000 lookups
                for ( int i = 0 ; i < 10 ; i++ ) { 
                    obj.lookupOperation("t1.txt");                   
                }
                System.out.println(mc.gpuTime("cuda_profile_0.log")/1000 + "," + time);
            }
            double gpuTime = mc.gpuTime("cuda_profile_0.log");
            if ( isFirstRun == true ) {
                isFirstRun = false;
                firstGpu = gpuTime/1000;
                firstTime = time;               
            }
            totalTime = time + (gpuTime/1000);//gpu time is in micro seconds
            obj.displayResults(totalTime);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /*** GPU related methods section ***/
    
    void setupGPU() throws IOException {
       // Initialize GPU.
       gpu = Gpu.gpu();
       gpu.ensureComputeCapability (2, 0);
       
       module = gpu.getModule ("./ParallelLacfKernel.cubin");
       
       kernel = module.getKernel(KernelFunctionInterface.class);       
       kernel.setBlockDim (3);
       kernel.setGridDim (gpu.getMultiprocessorCount()); 
       kernel.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_SHARED);
       
       kernel1 = module.getKernel(KernelFunctionInterface1.class);
       kernel1.setBlockDim (3);
       kernel1.setGridDim (gpu.getMultiprocessorCount()); 
       kernel1.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_SHARED);
    }
    
    /**
     * Method move the filter to the MP shared memory
     */
    void moveFilterToGpu() throws IOException {
      //transfer to GPU global memory first
        GpuStructArray<FilterEntry> gpuFilter = 
                gpu.getStructArray(FilterEntry.class, fsize);
        for ( int i = 0 ; i < fsize ; i++ ) {
            gpuFilter.item[i] = filter[i];
        }
        gpuFilter.hostToDev();

        //then transfer to shared memory        
        kernel.parallelLookup(gpuFilter);
    }
    
    /**
     * Specify that this task requires one core.
     */
    protected static int coresRequired() {
        return 1;
    }

    /**
     * Specify that this task requires one GPU accelerator.
     */
    protected static int gpusRequired() {
        return 1;
    }    
}
