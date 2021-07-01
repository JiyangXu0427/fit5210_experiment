/*******************************************************************************
 * Copyright (C) 2015 Francois Petitjean
 * 
 * This file is part of Chordalysis.
 * 
 * Chordalysis is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * Chordalysis is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Chordalysis.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package core.stats.scorer;

import core.model.DecomposableModel;
import core.model.GraphAction;
import core.model.ScoredGraphAction;
import core.stats.MessageLengthFactorialComputer;

public class GraphActionScorerMML extends GraphActionScorer {

  MessageLengthFactorialComputer computer;
  @Deprecated
  public GraphActionScorerMML(int nbInstances,MessageLengthFactorialComputer computer){
    this.nbInstances = nbInstances;
    this.computer = computer;
  }

  public GraphActionScorerMML(MessageLengthFactorialComputer computer){
    this.computer = computer;
    this.nbInstances = computer.getNbInstances();
  }

  @Override
  public ScoredGraphAction scoreEdge(DecomposableModel model, GraphAction action) {

    double diffLength =model.messageLengthDiffIfAdding(action.getV1(),action.getV2(), computer, false);
    ScoredGraphAction scoredAction = new ScoredGraphAction(action.getType(),action.getV1(), action.getV2(), diffLength);
    return scoredAction;

  }

}
