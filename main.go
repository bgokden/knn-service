package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"strconv"
	"sync"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"

	"github.com/bgokden/go-kdtree"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/testdata"

	pb "github.com/bgokden/knn-service/knnservice"
)

var (
	tls        = flag.Bool("tls", false, "Connection uses TLS if true, else plain TCP")
	certFile   = flag.String("cert_file", "", "The TLS cert file")
	keyFile    = flag.String("key_file", "", "The TLS key file")
	jsonDBFile = flag.String("json_db_file", "testdata/route_guide_db.json", "A json file containing a list of features")
	port       = flag.Int("port", 10000, "The server port")
)

type knnServiceServer struct {
	pointsMu sync.RWMutex   // protects points
	points   []kdtree.Point // read-only after initialized
	treeMu   sync.RWMutex   // protects KDTree
	tree     *kdtree.KDTree
}

type EuclideanPoint struct {
	kdtree.PointBase
	timestamp int64
	label     string
}

// Return the label
func (p *EuclideanPoint) GetLabel() string {
	return p.label
}

// Return the timestamp
func (p *EuclideanPoint) GetTimestamp() int64 {
	return p.timestamp
}

func (p *EuclideanPoint) Distance(other kdtree.Point) float64 {
	var ret float64
	for i := 0; i < p.Dim(); i++ {
		tmp := p.GetValue(i) - other.GetValue(i)
		ret += tmp * tmp
	}
	return ret
}

func (p *EuclideanPoint) PlaneDistance(val float64, dim int) float64 {
	tmp := p.GetValue(dim) - val
	return tmp * tmp
}

func NewEuclideanPoint(vals ...float64) *EuclideanPoint {
	ret := &EuclideanPoint{
		PointBase: kdtree.NewPointBase(vals),
	}
	return ret
}

func NewEuclideanPointWithLabel(timestamp int64, label string, vals ...float64) *EuclideanPoint {
	ret := &EuclideanPoint{
		PointBase: kdtree.NewPointBase(vals),
		timestamp: timestamp,
		label:     label,
	}
	return ret
}

func NewEuclideanPointArr(vals []float64) *EuclideanPoint {
	ret := &EuclideanPoint{
		PointBase: kdtree.NewPointBase(vals),
	}
	return ret
}

func NewEuclideanPointArrWithLabel(vals []float64, timestamp int64, label string) *EuclideanPoint {
	ret := &EuclideanPoint{
		PointBase: kdtree.NewPointBase(vals),
		timestamp: timestamp,
		label:     label,
	}
	return ret
}

func equal(p1 kdtree.Point, p2 kdtree.Point) bool {
	for i := 0; i < p1.Dim(); i++ {
		if p1.GetValue(i) != p2.GetValue(i) {
			return false
		}
	}
	return true
}

// CreateCustomer creates a new Customer
func (s *knnServiceServer) GetKnn(ctx context.Context, in *pb.KnnRequest) (*pb.KnnResponse, error) {
	point := NewEuclideanPointArr(in.GetFeature())
	s.treeMu.RLock()
	ans := s.tree.KNN(point, int(in.GetK()))
	s.treeMu.RUnlock()
	responseFeatures := make([]*pb.Feature, 0)
	for i := 0; i < len(ans); i++ {
		log.Println(ans[i])
		feature := &pb.Feature{
			Feature:   ans[i].GetValues(),
			Timestamp: ans[i].GetTimestamp(),
			Label:     ans[i].GetLabel(),
		}
		responseFeatures = append(responseFeatures, feature)
	}
	return &pb.KnnResponse{Id: in.Id, Features: responseFeatures}, nil
}

func (s *knnServiceServer) Insert(ctx context.Context, in *pb.InsertionRequest) (*pb.InsertionResponse, error) {
	point := NewEuclideanPointArrWithLabel(in.GetFeature(), in.GetTimestamp(), in.GetLabel())
	s.pointsMu.Lock()
	s.points = append(s.points, point)
	s.pointsMu.Unlock()
	s.pointsMu.RLock()
	s.treeMu.Lock()
	s.tree = kdtree.NewKDTree(s.points)
	s.treeMu.Unlock()
	s.pointsMu.RUnlock()
	return &pb.InsertionResponse{Id: in.Id, Code: 0}, nil
}

func newServer() *knnServiceServer {
	p1 := NewEuclideanPointWithLabel(time.Now().Unix(), "p1", 0.0, 0.0, 0.0)
	p2 := NewEuclideanPointWithLabel(time.Now().Unix(), "p2", 0.0, 0.0, 1.0)
	p3 := NewEuclideanPointWithLabel(time.Now().Unix(), "p3", 0.0, 1.0, 0.0)
	p4 := NewEuclideanPointWithLabel(time.Now().Unix(), "p4", 1.0, 0.0, 0.0)
	points := make([]kdtree.Point, 0)
	points = append(points, p1)
	points = append(points, p2)
	points = append(points, p3)
	points = append(points, p4)
	tree := kdtree.NewKDTree(points)
	s := &knnServiceServer{tree: tree}
	return s
}

func callKnn(client pb.KnnServiceClient) {
	request := &pb.KnnRequest{
		Id:        101,
		Timestamp: 1233234,
		Timeout:   5,
		K:         3,
		Feature: []float64{
			0.0,
			0.0,
			0.0,
		},
	}
	resp, err := client.GetKnn(context.Background(), request)
	if err != nil {
		log.Fatalf("There is an error: %v", err)
	}
	// if resp.Success {
	log.Printf("A new Response has been received with id: %d", resp.Id)
	features := resp.GetFeatures()
	for i := 0; i < len(features); i++ {
		log.Println(features[i].GetLabel())
	}
	// }
}

func callInsert(client pb.KnnServiceClient) {
	request := &pb.InsertionRequest{
		Id:        102,
		Timestamp: time.Now().Unix(),
		Label:     "po" + strconv.FormatInt(int64(rand.Intn(100)), 10),
		Feature: []float64{
			rand.Float64(),
			rand.Float64(),
			rand.Float64(),
		},
	}
	resp, err := client.Insert(context.Background(), request)
	if err != nil {
		log.Fatalf("There is an error: %v", err)
	}
	// if resp.Success {
	log.Printf("A new Response has been received with id: %d , code: %d", resp.Id, resp.Code)
	// }
}

func check() {
	for {
		// var opts []grpc.DialOption
		// var serverAddr string = "localhost:10000"
		conn, err := grpc.Dial("localhost:10000", grpc.WithInsecure())
		if err != nil {
			log.Fatalf("fail to dial: %v", err)
		}
		defer conn.Close()
		client := pb.NewKnnServiceClient(conn)

		callKnn(client)

		callInsert(client)

		// do some job
		log.Println(time.Now().UTC())
		time.Sleep(1000 * time.Millisecond)
	}
}

func main() {
	flag.Parse()
	lis, err := net.Listen("tcp", fmt.Sprintf("localhost:%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	var opts []grpc.ServerOption
	if *tls {
		if *certFile == "" {
			*certFile = testdata.Path("server1.pem")
		}
		if *keyFile == "" {
			*keyFile = testdata.Path("server1.key")
		}
		creds, err := credentials.NewServerTLSFromFile(*certFile, *keyFile)
		if err != nil {
			log.Fatalf("Failed to generate credentials %v", err)
		}
		opts = []grpc.ServerOption{grpc.Creds(creds)}
	}
	grpcServer := grpc.NewServer(opts...)
	pb.RegisterKnnServiceServer(grpcServer, newServer())
	go check()
	log.Println("Server started")
	grpcServer.Serve(lis)
}
