package comp9313.ass4;


import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
/* 
 * Tianyi Zhu
 * z5097194
 */
public class SetSimJoin {

	public static String OUT = "output";
	public static String IN = "input";
	
	// using RID pairs for secondary sort in stage 3 
	public static class RIDPair implements WritableComparable<RIDPair> {
		private int RID1;
		private int RID2;

		public RIDPair() {
			RID1 = 0;
			RID2 = 0;
		}

		public RIDPair(int RID1, int RID2) {
			set(RID1, RID2);
		}

		public void set(int left, int right) {
			RID1 = left;
			RID2 = right;
		}

		@Override
		public void readFields(DataInput in) throws IOException {
			// TODO Auto-generated method stub
			RID1 = in.readInt();
			RID2 = in.readInt();
		}

		@Override
		public void write(DataOutput out) throws IOException {
			// TODO Auto-generated method stub
			out.writeInt(RID1);
			out.writeInt(RID2);
		}

		public int getRID1() {
			return RID1;
		}

		public int getRID2() {
			return RID2;
		}

		// firstly compare first RID a, if equal, compare second RID
		@Override
		public int compareTo(RIDPair arg0) {
			// TODO Auto-generated method stub
			int cmp = RID1 - arg0.getRID1();
			if (cmp != 0) {
				return cmp;
			} else {
				return RID2 - arg0.getRID2();
			}
		}

	}
	// GroupComparator for the first sort in each part 
	public static class GroupComparator extends WritableComparator {
		protected GroupComparator() {

			super(RIDPair.class, true);
		}

		@Override
		public int compare(WritableComparable w1, WritableComparable w2) {
			RIDPair RID1 = (RIDPair) w1;
			RIDPair RID2 = (RIDPair) w2;
			return RID1.getRID1() - RID2.getRID1();
		}
	}
	
	// mapper in stage 2
	public static class Stage_2_Mapper extends Mapper<Object, Text, IntWritable, Text> {

		private Double T;
		// get parameter T before mapping
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);

			try {
				Configuration conf = context.getConfiguration();
				T = Double.parseDouble(conf.get("T"));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			
			// split each line to a list
			StringTokenizer itr = new StringTokenizer(value.toString(), " ");
			String RID = "";
			String value_sequence = "";

			// get first one as RID
			if (itr.hasMoreTokens()) {
				RID = itr.nextToken();

			}
			value_sequence = RID + ",";
			// the rest of it is records list ,get length of it first
			int len = itr.countTokens();
			// calculate prefix length
			int prefix;
			if (len * T % 1 == 0)
				prefix = len - ((int) (len * T)) + 1;
			else
				prefix = len - ((int) (len * T) + 1) + 1;

			int[] pre_tokens = new int[prefix];
			int i = 0;
			// save records to value_sequence and save prefix list to pre_tokens
			while (itr.hasMoreTokens()) {
				String element = itr.nextToken();
				value_sequence = value_sequence + element + " ";
				if (i < prefix) {
					pre_tokens[i] = Integer.parseInt(element);
				}
				i++;

			}
			// travel all of prefix list, write (each prefix, records list)
			for (int j = 0; j < prefix; j++) {
				context.write(new IntWritable(pre_tokens[j]), new Text(value_sequence));
			}

		}
	}
	
	// reducer for stage 2
	public static class Stage_2_Reducer extends Reducer<IntWritable, Text, Text, DoubleWritable> {

		private Double T;
		// get parameter T 
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);

			try {
				Configuration conf = context.getConfiguration();
				T = Double.parseDouble(conf.get("T"));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		@Override
		public void reduce(IntWritable key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {

			// using Hashmap to store all tokens (including RID and record list) with same prefix records
			HashMap<Integer, HashSet<Integer>> token_list = new HashMap<>();
			// suing ArrayList to store RIDs of all tokens
			ArrayList<Integer> RID_list = new ArrayList<Integer>();
			
			for (Text val : values) {
				// using hashset to store records in each token
				HashSet<Integer> records = new HashSet<>();

				StringTokenizer itr = new StringTokenizer(val.toString(), " ,");
				String RID = "";

				if (itr.hasMoreTokens()) {
					RID = itr.nextToken();
					RID_list.add(Integer.parseInt(RID));
				}

				while (itr.hasMoreTokens()) {
					String element = itr.nextToken();
					records.add(Integer.parseInt(element));
				}
				token_list.put(Integer.parseInt(RID), records);

			}
			// get number of tokens
			int len = RID_list.size();
			// compare tokens with each other ,finding similarity >= T
			for (int i = 0; i < len; i++) {
				double sim = 0;
				for (int j = i + 1; j < len; j++) {
					HashSet<Integer> token_1 = token_list.get(RID_list.get(i));
					HashSet<Integer> token_2 = token_list.get(RID_list.get(j));
					
					HashSet<Integer> Intersection = (HashSet<Integer>) token_1.clone();
					Intersection.retainAll(token_2);
					HashSet<Integer> union = (HashSet<Integer>) token_1.clone();
					union.addAll(token_2);
					// similarity is length of Intersection of A and B divides length of union of A and B
					sim = Intersection.size() / (union.size() + .0);

					// write ones satisfies sim >= T
					if (sim >= T) {
						context.write(new Text(RID_list.get(i) + " " + RID_list.get(j)), new DoubleWritable(sim));
					}
				}
			}
		}
	}

	// mapper for stages 3
	public static class Stage_3_Mapper extends Mapper<Object, Text, RIDPair, Text> {
		@Override

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			
			
			StringTokenizer itr = new StringTokenizer(value.toString(), " \t");
			// get first, second RID, and similarity of each lines
			int RID1 = Integer.parseInt(itr.nextToken());
			int RID2 = Integer.parseInt(itr.nextToken());
			String sim = itr.nextToken();
			// using RIDPair as key to write
			RIDPair pair = new RIDPair();
			if (RID1 > RID2) {

				pair.set(RID2, RID1);
				context.write(pair, new Text(sim));
			} else {
				pair.set(RID1, RID2);
				context.write(pair, new Text(sim));
			}
		}
	}
	// partitioner for secondary sort
	public static class Stage_3_Partitioner extends Partitioner<RIDPair, Text> {

		@Override
		public int getPartition(RIDPair key, Text txt, int numberOfPartitions) {
			return Math.abs(key.getRID1() % numberOfPartitions);
		}
	}
	
	// reducer for stage 3
	public static class Stage_3_Reducer extends Reducer<RIDPair, Text, Text, Text> {
		@Override
		public void reduce(RIDPair key, Iterable<Text> value, Context context)
				throws IOException, InterruptedException {
			String out = "";
//			System.out.println(key.getRID1()+"`````````"+key.getRID2());
//			int RID1 = key.getRID1();
			int RID2 = -1;
			// Hadoop using first RID in pair as key for reduce, get each second RID and sim
			for (Text val : value) {
//				System.out.println(key.getRID1()+"`````````"+key.getRID2());
				out = val.toString();
				// if token(RID1,RID2) has not been written, write it
				if (RID2!=key.getRID2()){
					context.write(new Text("(" + key.getRID1() + "," + key.getRID2() + ")"), new Text(out));
				}
				RID2 = key.getRID2();

			}
			

		}
	}

	public static void main(String[] args) throws Exception {
		long start, mid, end;
		start = System.currentTimeMillis();

		IN = args[0];
		OUT = args[1];
		Double T = Double.parseDouble(args[2]);
		int num_of_reducer = Integer.parseInt(args[3]);

		Configuration conf = new Configuration();
		conf.set("T", Double.toString(T));


		String input = IN;
		String output = OUT + "TEMP";

		//start a job for stage 2
		Job Stage_2_job = Job.getInstance(conf, "Stage_2_Job");

		Stage_2_job.setJarByClass(SetSimJoin.class);

		Stage_2_job.setMapperClass(Stage_2_Mapper.class);
		Stage_2_job.setReducerClass(Stage_2_Reducer.class);
		Stage_2_job.setNumReduceTasks(num_of_reducer);

		Stage_2_job.setMapOutputKeyClass(IntWritable.class);
		Stage_2_job.setMapOutputValueClass(Text.class);
		Stage_2_job.setOutputKeyClass(Text.class);
		Stage_2_job.setOutputValueClass(DoubleWritable.class);

		FileInputFormat.addInputPath(Stage_2_job, new Path(input));
		FileOutputFormat.setOutputPath(Stage_2_job, new Path(output));
		Stage_2_job.waitForCompletion(true);
		
		mid = System.currentTimeMillis();

		input = output;
		output = OUT;
		// start a job for stage 3
		Job Stage_3_job = Job.getInstance(conf, "Stage_3_Job");
		Stage_3_job.setJarByClass(SetSimJoin.class);

		Stage_3_job.setMapperClass(Stage_3_Mapper.class);
		//using Partitioner and Comparator to implement secondary sort
		Stage_3_job.setPartitionerClass(Stage_3_Partitioner.class);
		Stage_3_job.setGroupingComparatorClass(GroupComparator.class);
		Stage_3_job.setReducerClass(Stage_3_Reducer.class);

		Stage_3_job.setNumReduceTasks(num_of_reducer);

		Stage_3_job.setMapOutputKeyClass(RIDPair.class);
		Stage_3_job.setMapOutputValueClass(Text.class);
		Stage_3_job.setOutputKeyClass(Text.class);
		Stage_3_job.setOutputValueClass(Text.class);

		FileInputFormat.addInputPath(Stage_3_job, new Path(input));
		FileOutputFormat.setOutputPath(Stage_3_job, new Path(output));
		Stage_3_job.waitForCompletion(true);

		end = System.currentTimeMillis();
//		System.out.println(mid - start);
//		System.out.println(end - mid);

	}
}
