import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.ArrayList;

public class RouteLookup {
	
	public HashMap<Long, String> prefixTable;
	public volatile HashMap<Integer, HashMap<Long, String>> routeTable;
	public int successCounter;
	public int failureCounter;
	public int hashProbe, popPrefixCounter=0;
	
	public RouteLookup() {
		routeTable = new HashMap<>();
		hashProbe = 0;
	}
	
	public void insert(int prefixLength, long prefix) {
		
		if(routeTable.get(prefixLength) == null) {
			prefixTable = new HashMap<>();
			prefixTable.put(prefix, "nexthop");
			routeTable.put(prefixLength, prefixTable);
		} else {
			routeTable.get(prefixLength).put(prefix, "nexthop");
		}
	}

	public boolean lookup(long ip) {
		
		int count = 0;
		
		for(int index = 32; index >= 1; index--) {
			hashProbe++;
			count++;
			if(routeTable.get(index) != null) {
				if(routeTable.get(index).get(ip) != null) {
				    if ( (index > 14) && (index < 24) )
				        popPrefixCounter++;
					return true;
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
	
	public static void main(String[] args) throws IOException {
		
		RouteLookup rlookup = new RouteLookup();		
		String fileName = "p1.txt";
		File fin = new File(fileName);
		BufferedReader br  = new BufferedReader(new FileReader(fin));
		ArrayList<String> testData = new ArrayList<String>();
		String line = null;
		int count = 0;
		while((line = br.readLine())!= null) {
			line = line.replace("\n", "");
			String info[] = line.split("/");
			int prefixLength = Integer.parseInt(info[1]);	
			long prefix = rlookup.convertIptoDecimal(info[0]);
			prefix = prefix >> (32 - prefixLength);	  
			rlookup.insert(prefixLength, prefix);
			count += 1;			
		}		
		int lcount = 0;
		System.out.println("Total read : " + count);
//		fileName = "t1.txt";
		fileName = "/Users/satyajeet/cs/capstone/lacf/dataset/dataset1/t1.txt";
		fin = new File(fileName);
		br  = new BufferedReader(new FileReader(fin));
		line = null;
		while((line = br.readLine())!= null) {
			lcount++;
			line = line.replace("\n", "");
			testData.add(line);
		}
		
		long ip;
		Clock t = new Clock();
        t.reset();
        t.start();
        for ( int j = 0 ; j < 1 ; j++ ) {
            for ( int k = 0 ; k < 1 ; k++ ) {
                for ( int i = 0 ; i < testData.size() ; i++ ) {     
                    ip = rlookup.convertIptoDecimal(testData.get(i));
                    if(rlookup.lookup(ip))
                        rlookup.successCounter += 1;
                    else 
                        rlookup.failureCounter += 1;
                }
            }
        }		
		t.stop();
        System.out.println("Lookup time: " + t.getTime());
		double avg = ((double) rlookup.hashProbe)/((double)lcount);
//		System.out.println("Average hash probe : " + avg);
		System.out.println("Success : " + rlookup.successCounter);
		System.out.println("Failure : " + rlookup.failureCounter);			
		System.out.println("Entries that are popular: " + rlookup.popPrefixCounter );
	}

}