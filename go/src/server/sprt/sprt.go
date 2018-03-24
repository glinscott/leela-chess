/*
    This file is part of Leela Chess.
    Leela Chess is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    Leela Chess is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
	along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
	
	This File is adapated from Cute Chess, see <https://github.com/cutechess/cutechess/>
*/

package sprt

import "math"

type SprtResult int

const (
	Continue  SprtResult = 0
	AcceptH0  SprtResult = 1
	AcceptH1  SprtResult = 2
)

type BayesElo struct {
	m_bayesElo float64
	m_drawElo float64
}

type SprtProbability struct {
 	m_pWin  float64
 	m_pLoss float64
 	m_pDraw float64
}

type SprtStatus struct {
	result SprtResult
	llr    float64
	lBound float64
	uBound float64
}

type Sprt struct {
	m_elo0   float64
	m_elo1   float64
	m_alpha  float64
	m_beta   float64
	m_wins   int
	m_losses int
	m_draws  int
}

func (Bayes *BayesElo) setElo(p SprtProbability) {
	Bayes.m_bayesElo = 200.0 * math.Log10(p.m_pWin / p.m_pLoss *
					(1.0 - p.m_pLoss) / (1.0 - p.m_pWin))
	Bayes.m_drawElo  = 200.0 * math.Log10((1.0 - p.m_pLoss) / p.m_pLoss *
					(1.0 - p.m_pWin) / p.m_pWin)
}

func (Bayes BayesElo) getBayesElo() float64 {
	return Bayes.m_bayesElo;
}

func (Bayes BayesElo) getDrawElo() float64 {
	return Bayes.m_drawElo;
}

func (Bayes BayesElo) getScale() float64 {
	x := math.Pow(10.0, - Bayes.m_drawElo / 400.0);
	return 4.0 * x / ((1.0 + x) * (1.0 + x));
}

func (Prob *SprtProbability) setProb(wins int, losses int, draws int) { 
	count := wins + losses + draws;

    Prob.m_pWin = float64(wins) / float64(count)
    Prob.m_pLoss = float64(losses) / float64(count)
    Prob.m_pDraw = 1.0 - Prob.m_pWin - Prob.m_pLoss
}

func (Prob *SprtProbability) setProb2(b BayesElo) {
    Prob.m_pWin  = 1.0 / (1.0 + math.Pow(10.0, (b.getDrawElo() - b.getBayesElo()) / 400.0));
    Prob.m_pLoss = 1.0 / (1.0 + math.Pow(10.0, (b.getDrawElo() + b.getBayesElo()) / 400.0));
    Prob.m_pDraw = 1.0 - Prob.m_pWin - Prob.m_pLoss;
}

func MakeSprt(
	elo0 float64,
	elo1 float64,
	alpha float64, 
	beta float64,
	wins int,
    losses int,
    draws int) Sprt {
		return Sprt{elo0, elo1, alpha, beta, wins, losses, draws}
}

func (Status SprtStatus) GetLlr() float64 {
	return Status.llr
}

func (Status SprtStatus) GetResult() SprtResult {
	return Status.result
}

func (Sprt Sprt) GetStatus() SprtStatus {
	status := SprtStatus{Continue, 0.0, 0.0, 0.0}

	if (Sprt.m_wins <= 0 || Sprt.m_losses <= 0 || Sprt.m_draws <= 0) {
		return status
	}
	// Estimate draw_elo out of sample
	p := SprtProbability{0.0, 0.0, 0.0}
	p.setProb(Sprt.m_wins, Sprt.m_losses, Sprt.m_draws)
	b := BayesElo{0.0, 0.0}
    b.setElo(p)

	// Probability laws under H0 and H1
	s := b.getScale()
	b0 := BayesElo{Sprt.m_elo0 / s, b.m_drawElo}
	b1 := BayesElo{Sprt.m_elo1 / s, b.m_drawElo}
	p0 := SprtProbability{0.0, 0.0, 0.0}
	p0.setProb2(b0)
	p1 := SprtProbability{0.0, 0.0, 0.0}
	p1.setProb2(b1)

	// Log-Likelyhood Ratio
	status.llr = float64(Sprt.m_wins) * math.Log(p1.m_pWin / p0.m_pWin) +
	             float64(Sprt.m_losses) * math.Log(p1.m_pLoss / p0.m_pLoss) +
	             float64(Sprt.m_draws) * math.Log(p1.m_pDraw / p0.m_pDraw);

	// Bounds based on error levels of the test
	status.lBound = math.Log(Sprt.m_beta / (1.0 - Sprt.m_alpha));
	status.uBound = math.Log((1.0 - Sprt.m_beta) / Sprt.m_alpha);

	if (status.llr > status.uBound){
		status.result = AcceptH1
	} else if (status.llr < status.lBound){
		status.result = AcceptH0
	}

	return status;
}