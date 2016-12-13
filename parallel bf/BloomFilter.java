import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.TreeSet;

import edu.rit.pj2.Task;
import edu.rit.gpu.Kernel;
import edu.rit.gpu.CacheConfig;
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuIntArray;
import edu.rit.gpu.Module;

interface KernelFunctionInterface extends Kernel {
    public void parallelLookup(GpuIntArray entry);    
}

interface KernelFunctionInterface1 extends Kernel {
    public void parallelLookup1(GpuIntArray md5,
                                GpuIntArray sha,
                                GpuIntArray result); 
}

public class BloomFilter extends Task {

    Gpu gpu;
    Module module;
    KernelFunctionInterface kernel;
    KernelFunctionInterface1 kernel1;    
    
    final int fsize = 4000;    
    static int successCnt=0, failureCnt=0;//lookup counter
    int[] filter;
    public HashMap<Long, String> prefixTable;
    public HashMap<Integer, HashMap<Long, String>> routeTable;
    static double lookupTime;
    int hashProbes;
    
    void init() {
        routeTable = new HashMap<>();
        filter = new int[fsize];
        lookupTime = 0;
        hashProbes = 0;
    }
    
    /**
     * Helper functions 
     */
    
    public long convertIptoDecimal(String info) {    
        String ip[] = info.split("\\.");
        
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < ip.length; i++) {
            sb.append(String.format("%02X", Integer.parseInt(ip[i])));  
        }
        String pre = sb.toString();
        Long prefix = Long.parseLong(pre, 16);
        return prefix;
    }
    
    String hashString(String message, String algorithm) {
        
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
    
    String convertByteArrayToHexString(byte[] arrayBytes) {
        StringBuffer stringBuffer = new StringBuffer();
        for (int i = 0; i < arrayBytes.length; i++) {
            stringBuffer.append(Integer.toString((arrayBytes[i] & 0xff) + 0x100, 16)
                    .substring(1));
        }
        return stringBuffer.toString();
    }
    
    void displayResults(double totalTime) {
        System.out.println("Success count: " + successCnt);
        System.out.println("Failure count: " + failureCnt);
        System.out.println("Bloom final avg: " + totalTime/25);
    }
    
    /**
     * creates the bloom filter from prefix entries in fileName
     * @param fileName
     */
    void create(String fileName) {
        try {
            int mappedValue;
            String line, hash;
            BigInteger val;
            BufferedReader br  = 
                    new BufferedReader(new FileReader(new File(fileName)));
            while((line = br.readLine())!= null) {
                line = line.replace("\n", "");
                String info[] = line.split("/");
                int prefixLength = Integer.parseInt(info[1]);
                long prefix = convertIptoDecimal(info[0]);
                prefix = prefix >> (32 - prefixLength);
        
                //first insert in actual routing table
                if(routeTable.get(prefixLength) == null) {
                    prefixTable = new HashMap<>();
                    prefixTable.put(prefix, "nexthop");
                    routeTable.put(prefixLength, prefixTable);
                } else {
                    routeTable.get(prefixLength).put(prefix, "nexthop");
                }
        
                //compute positions now
                hash = hashString(prefix+"", "MD5");                
                val = new BigInteger(hash, 16);
                mappedValue = val.intValue();                
                mappedValue = Math.abs(mappedValue % fsize);
                filter[mappedValue] = 1;
                
                hash = hashString(prefix+"", "SHA");                
                val = new BigInteger(hash, 16);
                mappedValue = val.intValue();                
                mappedValue = Math.abs(mappedValue % fsize);
                filter[mappedValue] = 1;
            }
            br.close();
        }
        catch ( Exception e ) {
            e.printStackTrace();
        }
    }
    
    /**
     * performs lookup of all IP's in fileName
     * @param fileName
     */
    void lookup(String fileName) { 
        Clock t1 = new Clock();
        String line, hash;
        BigInteger val;
        boolean found = false;
        int mappedValue1, mappedValue2;
        GpuIntArray md5 = gpu.getIntArray(33);
        GpuIntArray sha = gpu.getIntArray(33);
        GpuIntArray result = gpu.getIntArray(33);
        ArrayList<String> testData = new ArrayList<String>();
        
        try {
            BufferedReader br  = 
                    new BufferedReader(new FileReader(new File(fileName)));
            while((line = br.readLine())!= null) {
                line = line.replace("\n", "");
                testData.add(line);
            }
            br.close();
            
            for ( String l : testData ) {
                l = l.replace("\n", "");
                long ip = convertIptoDecimal(l);                
                for ( int index = 32 ; index >= 1 ; index-- ) {
                    hash = hashString(ip+"", "MD5");                
                    val = new BigInteger(hash, 16);
                    mappedValue1= val.intValue();         
                    mappedValue1 = Math.abs(mappedValue1 % fsize);
                    
                    hash = hashString(ip+"", "SHA");                
                    val = new BigInteger(hash, 16);
                    mappedValue2 = val.intValue();
                    mappedValue2 = Math.abs(mappedValue2 % fsize);
                    
                    md5.item[index] = mappedValue1;
                    sha.item[index] = mappedValue2;
                    
                    ip = ip >> 1;
                }
                md5.hostToDev();
                sha.hostToDev();
               
                kernel1.parallelLookup1(md5, sha, result);                
                result.devToHost();
               
                t1.reset();
                t1.start();
                //start looking from highest
                TreeSet<Integer> possibleLens = new TreeSet<Integer>(Collections.reverseOrder());
                for ( int i = 1 ; i < 33 ; i++ ) {
                    if ( result.item[i] != -1 ) {
                        possibleLens.add(result.item[i]);
                    }
                }                                   
                
                //now lookup hashtable/actual forwarding table
                Iterator<Integer> it = possibleLens.iterator();
                ip = convertIptoDecimal(l);
                while ( it.hasNext() ) {
                    int index = (int)it.next();
                    long tmpIp = ip >> (32 - index);
                    hashProbes++;
                    if ( routeTable.get(index) != null ) {
                        if ( routeTable.get(index).get(tmpIp) != null ) {
                            //have found it!                            
                            successCnt++;
                            found = true;
                            break;
                        }
                    }
                }            
                if ( found == false ) {                
                    failureCnt++;
                }
                found = false;
                t1.stop();
                lookupTime += t1.getTime();
            }            
        } catch ( Exception e ) {
            e.printStackTrace();
        }
    }

    public void main(String[] arg0) throws Exception {
        MeanCalculator mc = new MeanCalculator();
        BloomFilter bf = new BloomFilter();        
        double totalTime = 0;
        bf.init();
        bf.setupGPU();
        bf.create("p1.txt");
        bf.moveFilterToGpu();
        for ( int i = 0 ; i < 25 ; i++ ) {
            for ( int j = 0 ; j < 10 ; j++ ) {
                bf.lookup("t1.txt");
            }
            System.out.println(mc.gpuTime("cuda_profile_0.log")/1000 + "," + lookupTime);
        }
        double gpuTime = mc.gpuTime("cuda_profile_0.log");
//        totalTime = totalTime + lookupTime + (gpuTime/1000);
        bf.displayResults(gpuTime/1000);
    }
    
    /**
     * Method move the filter to the MP shared memory
     */
    void   moveFilterToGpu() throws IOException {
      //transfer to GPU global memory first
        GpuIntArray gpuFilter = 
                gpu.getIntArray(fsize);
        for ( int i = 0 ; i < fsize ; i++ ) {
            gpuFilter.item[i] = filter[i];
        }
        gpuFilter.hostToDev();

        //then transfer to shared memory
        kernel.parallelLookup(gpuFilter);
    }

    void setupGPU() throws IOException {
        // Initialize GPU.
        gpu = Gpu.gpu();
        gpu.ensureComputeCapability (2, 0);

        module = gpu.getModule ("./ParallelBfKernel.cubin");
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