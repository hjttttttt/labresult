import os
import re


def result_parser(result_path):
    """_summary_
    Args:
        result_path (str): _description_
    Returns:
        tuple[List[float], List[float], Dict]: _description_
    """
    # Initialize lists and dictionaries.
    acc_list = []
    losses_list = []
    setting_dict = {}

    # Read the contents of the txt file.
    with open(result_path, 'r') as f:
        content = f.read()

    # Match the acc and losses list.
    acc_pattern = re.compile(r"acc:(\[[\d.,\s]*\])")
    losses_pattern = re.compile(r"loss:(\[[\d.,\s]*\])")

    # Convert the matched string into a list type and store it in the corresponding list.
    acc_str = re.findall(acc_pattern, content)[0]
    acc_list = list(map(float, re.findall(r'\d+\.\d+', acc_str)))
    losses_str = re.findall(losses_pattern, content)[0]
    losses_list = list(map(float, re.findall(r'\d+\.\d+', losses_str)))

    # Match the configuration information.
    setting_pattern = re.compile(r"{(.+?)}", re.DOTALL)
    setting_str = re.findall(setting_pattern, content)[0]
    setting_str = re.sub(r'\n', '', setting_str)
    setting_str = re.sub(r'\s+', ', ', setting_str)
    setting_dict = eval(f"{{{setting_str}}}")

    return acc_list, losses_list, setting_dict


record_file = "result_record.txt"
    # check existence of record file
if os.path.exists(record_file):
    accs, losses_list, setting_dict_ = result_parser(record_file)

print(accs)
print(losses_list)
print(setting_dict_)





package mr

import "fmt"
import "log"
import "net/rpc"
import "hash/fnv"
import "os"
import "ioutil"


//
// Map functions return a slice of KeyValue.
//
type KeyValue struct {
	Key   string
	Value string
}

//
// use ihash(key) % NReduce to choose the reduce
// task number for each KeyValue emitted by Map.
//
func ihash(key string) int {
	h := fnv.New32a()
	h.Write([]byte(key))
	return int(h.Sum32() & 0x7fffffff)
}


//
// main/mrworker.go calls this function.
//
func Worker(mapf func(string, string) []KeyValue,
	reducef func(string, []string) string) {

	for{
		args := Args{}
		reply := Reply{}
		err := call("Coordinator.AssignTask", &args, &reply)
		if err != nil {
			fmt.Println("%v", err)
			time.Sleep(time.Second)
			continue
		}
		task := reply.Task
		switch task.TaskType {
		case MapTask:
			doMap(task, mapf)
		case ReduceTask:
			doReduce(task, reducef)
		default:
			log.Fatalf("Invalid task type: %d", task.Type)
		}
		args.taskId = task.taskId
		args.TaskType = task.TaskType
		err := call("Coordinator.TaskCompleted", &args, &reply)

	}

}	

//
// example function to show how to make an RPC call to the coordinator.
//
// the RPC argument and reply types are defined in rpc.go.
//
func CallExample() ExampleReply {

	// declare an argument structure.
	args := ExampleArgs{}

	// fill in the argument(s).
	args.X = 99

	// declare a reply structure.
	reply := ExampleReply{}

	// send the RPC request, wait for the reply.
	// the "Coordinator.Example" tells the
	// receiving server that we'd like to call
	// the Example() method of struct Coordinator.
	ok := call("Coordinator.Example", &args, &reply)
	if ok {
		// reply.Y should be 100.
		fmt.Printf("reply.Y %v\n", reply.Y)
	} else {
		fmt.Printf("call failed!\n")
	}
	return reply
}

//
// send an RPC request to the coordinator, wait for the response.
// usually returns true.
// returns false if something goes wrong.
//
func call(rpcname string, args interface{}, reply interface{}) bool {
	// c, err := rpc.DialHTTP("tcp", "127.0.0.1"+":1234")
	sockname := coordinatorSock()
	c, err := rpc.DialHTTP("unix", sockname)
	if err != nil {
		log.Fatal("dialing:", err)
	}
	defer c.Close()

	err = c.Call(rpcname, args, reply)
	if err == nil {
		return true
	}

	fmt.Println(err)
	return false
}


func doMap(task Task, mapf func(files string, content string) []KeyValue) {
	filename := task.filename
	file, err := os.Open(filename)   // 先打开文件
	if err != nil {
		log.Fatalf("cannot open %s, %v", filename, err)
	}
	content, err := ioutil.ReadAll(file)
	if err != nil {
		log.Fatalf("cannot read %v", filename)
	}
	defer file.Close()
	kva := mapf(filename, string(content))
	encoders := make([]*json.Encoder, task.nReduce)   //创建nReduce个中间文件,编码对象
	for i := 0; i < task.nReduce; i++ {
		intermediateFilename := intermediateFilename(task.jobName,task.taskId, i)   // taskId为map的编号，i为reduce的编号，最终会产生map个数*recude个数的文件
		intermediateFile, err := os.Create(intermediateFilename)
		if err != nil {
			log.Fatalf("cannot create %s, %v", intermediateFilename, err)
		}
		defer intermediateFile.Close()
		encoders[i] = json.NewEncoder(intermediateFile)
	}
	for _, kv := range kva {
		idx := ihash(kv.Key) % task.NReduce
		err := encoders[idx].Encode(&kv)
		if err != nil {
			log.Fatalf("Failed to encode intermediate data: %v", err)
		}
	}
}


func doReduce(task Task, reducef func(string, []string) string) {
	// read intermediate files
	intermediate := []KeyValue{}
	for i := 0; i < task.nMap; i++ {
		filename := intermediateFilename(task.jobName, i, task.taskId)    
		file, err := os.Open(filename)
		if err != nil {
			log.Fatalf("cannot open %s, %v", filename, err)
		}
		dec := json.NewDecoder(file)
		for {
			var kv KeyValue
			if err := dec.Decode(&kv); err != nil {
				break
			}
			intermediate = append(intermediate, kv)
		}
		file.Close()
	}

	// group by key
	sort.Slice(intermediate, func(i, j int) bool {
		return intermediate[i].Key < intermediate[j].Key
	})
	groups := make(map[string][]string)
	for _, kv := range intermediate {
		groups[kv.Key] = append(groups[kv.Key], kv.Value)
	}

	// write output
	outFilename := mergeName(task.jobName, task.taskId)
	outFile, err := os.Create(outFilename)
	if err != nil {
		log.Fatalf("cannot create %v", outFilename)
	}
	enc := json.NewEncoder(outFile)
	for key, values := range groups {
		output := reducef(key, values)
		if err := enc.Encode(KeyValue{key, output}); err != nil {
			log.Fatalf("cannot write %v", outFilename)
		}
	}
	outFile.Close()
}

func intermediateFilename(jobName string, mapID int, reduceID int) string {
	return "output." + jobName + "-intermediateFilename-" + strconv.Itoa(mapID) + "-" + strconv.Itoa(reduceID)
}

func mergeName(jobName string ,reduceID int) string {
	if reduceID < 0{
		return "mr-out-" + jobName
	}
	return "output." + jobName + "-mergeName-" + strconv.Itoa(reduceID)
}




















package main

//
// start a worker process, which is implemented
// in ../mr/worker.go. typically there will be
// multiple worker processes, talking to one coordinator.
//
// go run mrworker.go wc.so
//
// Please do not change this file.
//

import "6.5840/mr"
import "plugin"
import "os"
import "fmt"
import "log"

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "Usage: mrworker xxx.so\n")
		os.Exit(1)
	}
	fmt.Println("mrworker.go begin!!!")
	mapf, reducef := loadPlugin(os.Args[1])
	mr.Worker(mapf, reducef)
}

// load the application Map and Reduce functions
// from a plugin file, e.g. ../mrapps/wc.so
func loadPlugin(filename string) (func(string, string) []mr.KeyValue, func(string, []string) string) {
	p, err := plugin.Open(filename)
	if err != nil {
		log.Fatalf("cannot load plugin %v", filename)
	}
	xmapf, err := p.Lookup("Map")
	if err != nil {
		log.Fatalf("cannot find Map in %v", filename)
	}
	mapf := xmapf.(func(string, string) []mr.KeyValue)
	xreducef, err := p.Lookup("Reduce")
	if err != nil {
		log.Fatalf("cannot find Reduce in %v", filename)
	}
	reducef := xreducef.(func(string, []string) string)

	return mapf, reducef
}
