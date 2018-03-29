package sprt

import (
	"fmt"
	"math"
	"math/rand"
)

var bb = math.Log(10) / 400

const (
	H0       = 0
	H1       = 1
	Continue = 2
)

const (
	loss = -1
	draw = 0
	win  = 1
)

func L(x float64) float64 {
	return 1 / (1 + math.Exp(-bb*x))
}

// Computes the MLE for the draw ratio, subject to the condition score=s.
// This analytically correct formula was computed with Sage.
func DrawRatioMLE(s float64, results []float64) float64 {
	L := results[0]
	D := results[1]
	W := results[2]

	s2 := s * s
	W2 := W * W
	D2 := D * D
	L2 := L * L
	dr := (L*s - W*s + D + W - math.Sqrt(4*D2*s2+4*D*L*s2+4*D*W*s2+L2*s2-2*L*W*s2+
		W2*s2-4*D2*s-2*D*L*s-6*D*W*s+2*L*W*s-2*W2*s+D2+2*D*W+W2)) / (W + D + L)
	return dr
}

// Compute the log likelihood for score=s.
func LL(s float64, results []float64) float64 {
	d := DrawRatioMLE(s, results)
	w := s - d/2
	l := 1 - s - d/2
	L := results[0]
	D := results[1]
	W := results[2]
	return W*math.Log(w) + D*math.Log(d) + L*math.Log(l)
}

/***********************************************************************************************/

//A helper class for estimating the drift and infinitesimal variance
//of a Wiener process (sampled in discrete time).
type BrownianMLE struct {
	xl   float64
	mean float64
	M2   float64
	n    int
}

func add(b *BrownianMLE, x float64) {
	dx := x - b.xl
	b.xl = x
	// Wikipedia!
	b.n++
	delta := dx - b.mean
	b.mean += delta / float64(b.n)
	b.M2 += delta * (dx - b.mean)
}

func params(b *BrownianMLE) (float64, float64) {
	if b.n < 2 {
		return 0, 0 // adhoc for our application
	}

	return b.mean, math.Sqrt(b.M2 / float64(b.n-1))

}

/***********************************************************************************************/

//
//    This class performs a GSPRT for H0:elo=elo0 versus H1:elo=elo1
//    See here for a description of the GSPRT as well as theoretical (asymptotic) results.
//
//    http://stat.columbia.edu/~jcliu/paper/GSPRT_SQA3.pdf
//
//    To record the outcome of a game use the method record(result) where "result" is one of 0,1,2,
//    corresponding respectively to a loss, a draw and a win.
//
type SimpleSPRT struct {
	score0  float64
	score1  float64
	LA      float64
	LB      float64
	results [3]float64
	_status int
	b       *BrownianMLE
}

func NewSimpleSPRT(alpha float64, beta float64, elo0 float64, elo1 float64) *SimpleSPRT {
	eps := 0.01
	s := new(SimpleSPRT)
	s.results[0] = eps // small prior
	s.results[1] = eps
	s.results[2] = eps
	s.score0 = L(elo0)
	s.score1 = L(elo1)
	s.LA = math.Log(beta / (1 - alpha))
	s.LB = math.Log((1 - beta) / alpha)
	s._status = Continue
	s.b = new(BrownianMLE)
	return s
}

func (s SimpleSPRT) GetStatus() int {
	return s._status
}

func (s SimpleSPRT) AddRecord(result int) {
	var idx int

	if result == loss {
		idx = 0
	} else if result == draw {
		idx = 1
	} else if result == win {
		idx = 2
	} else {
		return
	}

	s.results[idx]++
	LLR := LL(s.score1, s.results[:]) - LL(s.score0, s.results[:])
	add(s.b, LLR)
	_, sigma := params(s.b)
	rho := 0.583 // for overshoot correction
	if LLR > s.LB-rho*sigma {
		s._status = H1
	} else if LLR < s.LA+rho*sigma {
		s._status = H0
	}
}

func status(s *SimpleSPRT) int {
	return s._status
}

/***********************************************************************************************/

func pick(w float64, d float64, l float64) int {
	s := rand.Float64()
	if s <= w {
		return 1
	} else if s <= w+d {
		return 0
	} else {
		return -1
	}
}

// Simulates the test H0:elo_diff=0 versus H1:elo_diff=epsilon with elo_diff equal
// to elo.
func simulate(drawRatio float64, epsilon float64, alpha float64, elo float64) int {
	sp := NewSimpleSPRT(alpha, alpha, 0, epsilon)
	s := L(elo)
	d := drawRatio
	w := s - d/2
	l := 1 - s - d/2
	for true {
		r := pick(w, d, l)
		sp.AddRecord(r)
		status := status(sp)
		if status != Continue {
			return status
		}
	}
	return Continue // We never get here
}

func main() {
	epsilon := 5.0
	a := [2]int{0, 0}
	cc := 0
	for true {
		status := simulate(0.33, epsilon, 0.05, 0)
		a[status]++
		cc++
		r0 := float64(a[H0]) / float64(cc)
		r1 := float64(a[H1]) / float64(cc)
		if cc%100 == 0 {
			fmt.Printf("%d %f %f +- %f\n", cc, r0, r1, 1.96*math.Pow(r0*(1-r0), 0.5)/math.Sqrt(float64(cc)))
		}
	}
}
