package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"

	"github.com/bgokden/go-kdtree"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/testdata"

	pb "github.com/bgokden/knn-service/knnservice"
	"github.com/segmentio/ksuid"
)

var (
	tls        = flag.Bool("tls", false, "Connection uses TLS if true, else plain TCP")
	certFile   = flag.String("cert_file", "", "The TLS cert file")
	keyFile    = flag.String("key_file", "", "The TLS key file")
	jsonDBFile = flag.String("json_db_file", "testdata/route_guide_db.json", "A json file containing a list of features")
	port       = flag.Int("port", 10000, "The server port")
)

const k = 3

type Peer struct {
	address string
	version string
	avg     []float64
}

type knnServiceServer struct {
	k         int64
	avg       []float64
	n         int64
	address   string
	version   string
	pointsMap sync.Map
	services  sync.Map
	peers     sync.Map
	pointsMu  sync.RWMutex   // protects points
	points    []kdtree.Point // read-only after initialized
	treeMu    sync.RWMutex   // protects KDTree
	tree      *kdtree.KDTree
}

type EuclideanPoint struct {
	kdtree.PointBase
	timestamp int64
	label     string
}

type EuclideanPointKey struct {
	feature   [k]float64
	timestamp int64
}

type EuclideanPointValue struct {
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

func NewEuclideanPointArrWithLabel(vals [k]float64, timestamp int64, label string) *EuclideanPoint {
	slice := make([]float64, k)
	copy(slice, vals[:])
	ret := &EuclideanPoint{
		PointBase: kdtree.NewPointBase(slice),
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
	if s.tree != nil {
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
	return &pb.KnnResponse{Id: in.Id, Features: nil}, nil
}

func (s *knnServiceServer) Insert(ctx context.Context, in *pb.InsertionRequest) (*pb.InsertionResponse, error) {
	key := EuclideanPointKey{
		timestamp: in.GetTimestamp(),
	}
	copy(key.feature[:], in.GetFeature())
	value := EuclideanPointValue{
		timestamp: in.GetTimestamp(),
		label:     in.GetLabel(),
	}
	s.pointsMap.Store(key, value)
	return &pb.InsertionResponse{Id: in.Id, Code: 0}, nil
}

func (s *knnServiceServer) Join(ctx context.Context, in *pb.JoinRequest) (*pb.JoinResponse, error) {
	p, _ := peer.FromContext(ctx)
	address := strings.Split(p.Addr.String(), ":")[0] + ":" + strconv.FormatInt(int64(in.GetPort()), 10)
	log.Printf("Peer Addr: %s", address)
	peer := Peer{
		address: address,
		avg:     in.GetAvg(),
		version: in.GetVersion(),
	}
	s.peers.Store(address, peer)
	return &pb.JoinResponse{Address: address}, nil
}

func (s *knnServiceServer) ExchangeServices(ctx context.Context, in *pb.ServiceMessage) (*pb.ServiceMessage, error) {
	inputServiceList := in.GetServices()
	for i := 0; i < len(inputServiceList); i++ {
		s.services.Store(inputServiceList[i], true)
	}
	outputServiceList := make([]string, 0)
	s.services.Range(func(key, value interface{}) bool {
		serviceName := key.(string)
		outputServiceList = append(outputServiceList, serviceName)
		return true
	})
	return &pb.ServiceMessage{Services: outputServiceList}, nil
}

func (s *knnServiceServer) ExchangePeers(ctx context.Context, in *pb.PeerMessage) (*pb.PeerMessage, error) {
	inputPeerList := in.GetPeers()
	for i := 0; i < len(inputPeerList); i++ {
		peer := Peer{
			address: inputPeerList[i].GetAddress(),
			version: inputPeerList[i].GetVersion(),
			avg:     inputPeerList[i].GetAvg(),
		}
		s.peers.Store(inputPeerList[i].GetAddress(), peer)
	}
	outputPeerList := make([]*pb.Peer, 0)
	s.peers.Range(func(key, value interface{}) bool {
		// address := key.(string)
		peer := value.(Peer)
		peerProto := &pb.Peer{
			Address: peer.address,
			Version: peer.version,
			Avg:     peer.avg,
		}
		outputPeerList = append(outputPeerList, peerProto)
		return true
	})
	return &pb.PeerMessage{Peers: outputPeerList}, nil
}

func (s *knnServiceServer) getClient(address string) (*pb.KnnServiceClient, *grpc.ClientConn) {
	conn, err := grpc.Dial(address, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("fail to dial: %v", err)
	}
	client := pb.NewKnnServiceClient(conn)
	return &client, conn
}

func (s *knnServiceServer) callJoin(client *pb.KnnServiceClient) {
	request := &pb.JoinRequest{
		Address: s.address,
		Avg:     s.avg,
		Port:    int32(*port),
		Version: s.version,
	}
	resp, err := (*client).Join(context.Background(), request)
	if err != nil {
		log.Printf("There is an error %v", err)
		return
	}
	if s.address != resp.GetAddress() {
		s.address = resp.GetAddress()
	}
}

func (s *knnServiceServer) callExchangeServices(client *pb.KnnServiceClient) {
	outputServiceList := make([]string, 0)
	s.services.Range(func(key, value interface{}) bool {
		serviceName := key.(string)
		outputServiceList = append(outputServiceList, serviceName)
		return true
	})
	request := &pb.ServiceMessage{
		Services: outputServiceList,
	}
	resp, err := (*client).ExchangeServices(context.Background(), request)
	if err != nil {
		log.Printf("There is an error %v", err)
		return
	}
	inputServiceList := resp.GetServices()
	for i := 0; i < len(inputServiceList); i++ {
		s.services.Store(inputServiceList[i], true)
	}
	log.Printf("Services exhanged")
}

func (s *knnServiceServer) callExchangePeers(client *pb.KnnServiceClient) {
	outputPeerList := make([]*pb.Peer, 0)
	s.peers.Range(func(key, value interface{}) bool {
		// address := key.(string)
		peer := value.(Peer)
		peerProto := &pb.Peer{
			Address: peer.address,
			Version: peer.version,
			Avg:     peer.avg,
		}
		outputPeerList = append(outputPeerList, peerProto)
		return true
	})

	request := &pb.PeerMessage{
		Peers: outputPeerList,
	}
	resp, err := (*client).ExchangePeers(context.Background(), request)
	if err != nil {
		log.Printf("There is an error %v", err)
		return
	}
	inputPeerList := resp.GetPeers()
	for i := 0; i < len(inputPeerList); i++ {
		peer := Peer{
			address: inputPeerList[i].GetAddress(),
			version: inputPeerList[i].GetVersion(),
			avg:     inputPeerList[i].GetAvg(),
		}
		s.peers.Store(inputPeerList[i].GetAddress(), peer)
	}
	log.Printf("Peers exhanged")
}

func (s *knnServiceServer) SyncJoin() {
	log.Printf("Sync Join")
	s.services.Range(func(key, value interface{}) bool {
		serviceName := key.(string)
		client, conn := s.getClient(serviceName)
		s.callJoin(client)
		conn.Close()
		return true
	})
	log.Printf("Service loop Ended")
	s.peers.Range(func(key, value interface{}) bool {
		peerAddress := key.(string)
		client, conn := s.getClient(peerAddress)
		s.callJoin(client)
		s.callExchangeServices(client)
		s.callExchangePeers(client)
		conn.Close()
		return true
	})
	log.Printf("Peer loop Ended")
}

func (s *knnServiceServer) syncMapToTree() {
	points := make([]kdtree.Point, 0)
	s.pointsMap.Range(func(key, value interface{}) bool {
		euclideanPointKey := key.(EuclideanPointKey)
		euclideanPointValue := value.(EuclideanPointValue)
		point := NewEuclideanPointArrWithLabel(
			euclideanPointKey.feature,
			euclideanPointKey.timestamp,
			euclideanPointValue.label)
		points = append(points, point)
		return true
	})
	if len(points) > 0 {
		tree := kdtree.NewKDTree(points)
		s.treeMu.Lock()
		s.tree = tree
		s.treeMu.Unlock()
	}
}

func newServer() *knnServiceServer {
	s := &knnServiceServer{}
	s.services.Store("localhost:10000", true)
	return s
}

func callKnn(client pb.KnnServiceClient) {
	id := ksuid.New()
	request := &pb.KnnRequest{
		Id:        id.String(),
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
	log.Printf("A new Response has been received with id: %s", resp.Id)
	features := resp.GetFeatures()
	for i := 0; i < len(features); i++ {
		log.Println(features[i].GetLabel())
	}
	// }
}

func callInsert(client pb.KnnServiceClient) {
	request := &pb.InsertionRequest{
		Id:        "102",
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
	log.Printf("A new Response has been received with id: %s , code: %d", resp.Id, resp.Code)
	// }
}

func (s *knnServiceServer) check() {
	for {
		// var opts []grpc.DialOption
		// var serverAddr string = "localhost:10000"
		conn, err := grpc.Dial("localhost:10000", grpc.WithInsecure())
		if err != nil {
			log.Fatalf("fail to dial: %v", err)
		}
		defer conn.Close()
		client := pb.NewKnnServiceClient(conn)

		s.SyncJoin()

		callKnn(client)

		callInsert(client)

		s.syncMapToTree()

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
	s := newServer()
	pb.RegisterKnnServiceServer(grpcServer, s)
	go s.check()
	log.Println("Server started")
	grpcServer.Serve(lis)
}
