import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.UnsupportedEncodingException;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;

public class CreateFilter {
    
    public static int fsize = 4000;
    public static int MAX_KICKS = 50;
    public FilterEntry filter[];
    public int prefixCount;
    
    CreateFilter() {
        filter = new FilterEntry[fsize];
        for(int i = 0; i < fsize; i++)            
            filter[i] = new FilterEntry();
    }
    
    public boolean insertAtPos(char fp, int pos) {
        if(filter[pos].size == 5) {
            return false;
        } else if(filter[pos].size == 0) {
            filter[pos].fingerprint[0] = (byte)fp;
            filter[pos].size += 1;
        } else {
            filter[pos].fingerprint[filter[pos].size] = (byte)fp;
            filter[pos].size += 1;
        }
        return true;
    }

    //just computes md5 of string and returns hex value
    private static String hashString(String message, String algorithm) {
     
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
    
    //converts byte array to hex string
    private static String convertByteArrayToHexString(byte[] arrayBytes) {
        StringBuffer stringBuffer = new StringBuffer();
        for (int i = 0; i < arrayBytes.length; i++) {
            stringBuffer.append(Integer.toString((arrayBytes[i] & 0xff) + 0x100, 16)
                    .substring(1));
        }
        return stringBuffer.toString();
    }
    
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

        //System.out.println("Get Position 2");
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
        
        //System.out.println("mapped value : " + mappedValue);
        //System.out.println("pos1 : " + pos1);
        
        int pos2 = mappedValue ^ pos1;
        pos2 = pos2 % fsize;
        //System.out.println("pos2" + pos2);
        return pos2;
        
    }
    
    //lenght aware ? 
    public boolean findPopularity(int prefixLength) {
        
        if(prefixLength >= 14 && prefixLength <= 24) {
            return true;
        } else {
            return false;
        }
        
//      return true;
    }

    
    public char generateFingerprint(String prefix) {
        String fp = hashString(prefix, "MD5");
        BigInteger val = new BigInteger(fp, 16);
        int mappedValue = val.intValue();
        mappedValue = (mappedValue & 0xFF) % 127;   
        char fingerprint = (char) mappedValue;
        return fingerprint;
    }
    
    public void insertAtFilter(int prefixLength, long prefix) {
        
        int pos1 = -1;
        int pos2 = -1;
        int pos11 = -1;
        int pos12 = -1;
        char fingerprint = generateFingerprint(Long.toString(prefix));
        
        //System.out.println("fp : " + fingerprint);
        
        pos1 = getPosition1(Long.toString(prefix), 1);
        pos2 = getPosition2(pos1, fingerprint, 1);

        if(findPopularity(prefixLength) == false) {
            pos11 = getPosition1(Long.toString(prefix), 2);
            pos12 = getPosition2(pos11, fingerprint, 2);
            //System.out.println(pos11);
            //System.out.println(pos12);
        }
        
        //System.out.println("Inserted at : " + pos1 + ", " + pos2 + ", " + pos11 + ", " + pos12);
        addItemLacf(fingerprint, pos1, pos2, pos11, pos12);
    }
    
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
    
    public boolean addItemLacf(char fingerprint, int pos1, int pos2, int pos11, int pos12) {

        if(addItem(fingerprint, pos1, pos2, 1) == true) {
            prefixCount += 1;
        } else {
            return false;
        }
        
        if(pos11 != -1 && pos12 != -1) {
            if(addItem(fingerprint, pos11, pos12, 2) == true) {
                prefixCount += 1;
            } else {
                return false;
            }
            
        }
        return true;
    }

    
    public void insert(int prefixLength, long prefix) {                
        insertAtFilter(prefixLength, prefix);
    }
    
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
    
    public void createFilter(String fileName) {
        File fin = new File(fileName);
        try {
            BufferedReader br  = new BufferedReader(new FileReader(fin));
            String line = null;
            int count = 0;
            
            while((line = br.readLine())!= null) {
                line = line.replace("\n", "");
                String info[] = line.split("/");
                //System.out.println("Inserting : " + info[0]);
                int prefixLength = Integer.parseInt(info[1]);
                long prefix = convertIptoDecimal(info[0]);
                prefix = prefix >> (32 - prefixLength);   
                insert(prefixLength, prefix);
                count += 1;
            }
        } catch (Exception e) {
            System.out.println(e);
        }
    }
    
    public static void main(String args[]) {
        CreateFilter createFilterObj = new CreateFilter();
        createFilterObj.createFilter("p1.txt");
        System.out.println((int)createFilterObj.filter[797].fingerprint[0]);
    }
}
