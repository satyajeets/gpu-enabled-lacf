import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;

public class RouteLookup {
	
	public FilterList filter[];
	public static int fsize = 4000;
	public static int MAX_KICKS = 50;
	public int hashProbe;
	public HashMap<Long, String> prefixTable;
	public HashMap<Integer, HashMap<Long, String>> routeTable;
	public int successCounter;
	public int failureCounter;
	public int prefixCount;
	public ArrayList<Integer> tmpValues = new ArrayList<Integer>();
	public boolean fillValues = false;
	
	public RouteLookup() {
		routeTable = new HashMap<>();
		filter = new FilterList[fsize];
		for(int i = 0; i < fsize; i++) {
			filter[i] = new FilterList();
		}
		hashProbe = 0;
		prefixCount = 0;
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
		if ( fillValues == true )
		    tmpValues.add(mappedValue);
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
		
//		return true;
	}
	
	public boolean insertAtPos(char fp, int pos) {
		if(filter[pos].size == 5) {
			return false;
		} else if(filter[pos].size == 0) {
			filter[pos].fingerprint[0] = fp;
			filter[pos].size += 1;
		} else {
			filter[pos].fingerprint[filter[pos].size] = fp;
			filter[pos].size += 1;
		}
		return true;
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
			char temp = filter[pos].fingerprint[4];
			filter[pos].fingerprint[4] = fingerprint;
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
	
	//filter insertion
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
	
	public void insert(int prefixLength, long prefix) {
		
		if(routeTable.get(prefixLength) == null) {
			prefixTable = new HashMap<>();
			prefixTable.put(prefix, "nexthop");
			routeTable.put(prefixLength, prefixTable);
		} else {
			routeTable.get(prefixLength).put(prefix, "nexthop");
		}
		
		insertAtFilter(prefixLength, prefix);
	}
	
	public boolean search(char fingerprint, int pos) {
		
		for(int i = 0; i < 5; i++) {
			if(filter[pos].fingerprint[i] == fingerprint) {
				return true;
			}
		}

		return false;
		
	}
	
	public boolean lookupPop(long prefix) {
		
		//System.out.println("Pop lookup");
		int pos1 = -1;
		int pos2 = -1;
		char fingerprint = generateFingerprint(Long.toString(prefix));
				
		pos1 = getPosition1(Long.toString(prefix), 1);
		pos2 = getPosition2(pos1, fingerprint, 1);
		
		//System.out.println("Fp = " + fingerprint + "pos1 = " + pos1 + "pos2 = " + pos2);
		if(search(fingerprint, pos1)) {
			return true;
		}
		
		if(search(fingerprint, pos2)) {
			return true;
		}
		
		return false;
	}
	
	public boolean lookupNonPop(long prefix) {
		
		//System.out.println("Non Pop lookup");
		int pos1 = -1;
		int pos2 = -1;
		char fingerprint = generateFingerprint(Long.toString(prefix));
		
		pos1 = getPosition1(Long.toString(prefix), 2);
		pos2 = getPosition2(pos1, fingerprint, 2);
		
		//System.out.println("Fp = " + fingerprint + "pos1 = " + pos1 + "pos2 = " + pos2);
		if(search(fingerprint, pos1)) {
			return true;
		}
		
		if(search(fingerprint, pos2)) {
			return true;
		}
		
		return false;
	}
	
	public boolean lookupLacf(int prefixLength, long ip) {
		boolean pop = findPopularity(prefixLength);
				
		if(pop == false) {
			if(lookupPop(ip) == true) {
				if(lookupNonPop(ip) == true) {
					return true;
				} else {
					return false;
				}
			} else {
				return false;
			}			
		} else {
			if(lookupPop(ip) == true) {
				return true;
			} else {
				return false;
			}
		}
	}
	
	public boolean lookup(long prefix) {
		
		long ip = prefix;
		int count = 0;
		
		for(int index = 32; index >= 1; index--) {
			
			if(lookupLacf(index, ip) == true) {
				hashProbe += 1;
				count += 1;
				if(routeTable.get(index) != null) {
					if(routeTable.get(index).get(ip) != null) {						    
						return true;
					}
				}
			}
			
			ip = ip >> 1;
		}
		
		hashProbe -= count;
		
		return false;
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
	
	void disp() {
	    for ( int i = 0 ; i < tmpValues.size() ; i++ )
	        System.out.println(tmpValues.get(i));
	}
	
	public static void main(String[] args) throws IOException {
		
		RouteLookup rlookup = new RouteLookup();
		String fileName = "prefix1.txt";
		File fin = new File(fileName);
		BufferedReader br  = new BufferedReader(new FileReader(fin));
		String line = null;
		int count = 0;
		ArrayList<String> testData = new ArrayList<String>();
		
		while((line = br.readLine())!= null) {
			line = line.replace("\n", "");
			String info[] = line.split("/");
			//System.out.println("Inserting : " + info[0]);
			int prefixLength = Integer.parseInt(info[1]);
			long prefix = rlookup.convertIptoDecimal(info[0]);
			prefix = prefix >> (32 - prefixLength);	  
			rlookup.insert(prefixLength, prefix);
			count += 1;
		}
		int lcount = 0;
		
		System.out.println("Total read : " + count);
		fileName = "lookup1.txt";
		fin = new File(fileName);
		br  = new BufferedReader(new FileReader(fin));
		line = null;
        
		while((line = br.readLine())!= null) {
			lcount += 1;
			line = line.replace("\n", "");
			testData.add(line);					
		}
		br.close();

		//start clock after the file i/o
		Clock t = new Clock();
        t.reset();
        t.start();

		rlookup.fillValues = true;
		for ( String l : testData ) {
		    long ip = rlookup.convertIptoDecimal(l);
            if(rlookup.lookup(ip)) {
                rlookup.successCounter += 1;
            } else {
//                System.out.println("Failed : " + l);
                rlookup.failureCounter += 1;
            }
		}

        t.stop();
        System.out.println("Lookup time: " + t.getTime());
		
		System.out.println("Success : " + rlookup.successCounter);
		System.out.println("Failure : " + rlookup.failureCounter);

//	    double avg = ((double) rlookup.hashProbe)/((double)lcount);
	    System.out.println("Average hash probe : " + rlookup.hashProbe);
//	    double tableOcc = (((double) rlookup.prefixCount) / (((double) fsize) * 5)) * 100;
//	    System.out.println("Table occupancy : " + tableOcc);
//        System.out.println(rlookup.hashProbe);		
	}
}
